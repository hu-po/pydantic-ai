import asyncio
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar, Generic, Self, TypeVar

import yaml
from pydantic import BaseModel, ValidationError
from pydantic_core import to_json, to_jsonable_python
from typing_extensions import TypeVar

from pydantic_ai import Agent, models

InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])


class TestCase(BaseModel, Generic[InputsT, OutputT, MetadataT]):
    name: str
    inputs: InputsT
    expected_output: OutputT
    metadata: MetadataT


class TestCases(BaseModel, Generic[InputsT, OutputT, MetadataT]):
    stem: ClassVar[str] = 'test_cases'  # subclass and override to change the stem used for the schema and yaml files

    test_cases: list[TestCase[InputsT, OutputT, MetadataT]]

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> Self:
        if path is None:
            path = cls._get_cases_path()

        path = Path(path)
        raw = path.read_text()
        loaded = yaml.safe_load(raw)
        return cls.model_validate(loaded)

    @classmethod
    def generate_cases_files(
        cls,
    ) -> None:
        cases_path = cls._get_cases_path()
        schema_ref = cls._get_schema_ref()
        schema_path = (cases_path.parent / schema_ref).resolve()

        schema_text = to_json(cls.model_json_schema(), indent=2).decode() + '\n'
        if not schema_path.exists() or schema_path.read_text() != schema_text:
            schema_path.write_text(schema_text)

        schema_ref = cls._get_schema_ref()
        yaml_language_server_line = f'# yaml-language-server: $schema={schema_ref}'
        if cases_path.exists():
            cases_text = cases_path.read_text()
            cases_text_with_schema = _ensure_yaml_language_server_line(cases_text, schema_ref)
            if cases_text_with_schema != cases_text:
                cases_path.write_text(cases_text_with_schema)
        else:
            content = yaml.dump(to_jsonable_python(cls(test_cases=[])), sort_keys=False)
            cases_path.write_text(f'{yaml_language_server_line}\n{content}')

    @classmethod
    def generate_examples(
        cls,
        model: models.Model | models.KnownModelName = 'gpt-4o',
        min_count: int = 3,
    ):
        existing_content: str | None = None
        cases_path = cls._get_cases_path()
        schema_ref = cls._get_schema_ref()

        if cases_path.exists():
            existing_cases = _load_existing_cases(cases_path, cls)
            min_count = max(0, min_count - len(existing_cases))
            if min_count == 0:
                return
            if existing_cases:
                existing_content = cases_path.read_text()

        examples = asyncio.run(_generate_examples(cls, cases_path, model, min_count))

        if existing_content is None:
            content = yaml.dump(to_jsonable_python(cls(test_cases=examples)), sort_keys=False)
            content = _ensure_yaml_language_server_line(content, schema_ref)
            cases_path.write_text(content)
        else:
            new_lines = yaml.dump(to_jsonable_python(cls(test_cases=examples)), sort_keys=False).splitlines()
            new_lines = new_lines[1:]  # drop the first line, which is the document start
            new_content = _ensure_yaml_language_server_line(existing_content, schema_ref)
            if not new_content.endswith('\n'):
                new_content += '\n'
            new_content += '\n'.join(new_lines)
            cases_path.write_text(new_content)

    @classmethod
    def _get_schema_ref(cls):
        return f'./{cls.stem}_schema.json'

    @classmethod
    def _get_cases_path(cls):
        module_path = sys.modules[cls.__module__].__file__
        assert isinstance(module_path, str), 'Module must be a file-based module'
        root = Path(module_path).parent
        return root / f'{cls.stem}.yaml'


def _ensure_yaml_language_server_line(content: str, schema_ref: str) -> str:
    first_line = content.split('\n', 1)[0]
    yaml_language_server_line = f'# yaml-language-server: $schema={schema_ref}'
    if first_line == yaml_language_server_line:
        return content
    elif first_line.startswith('# yaml-language-server: $schema='):
        return '\n'.join([yaml_language_server_line] + content.split('\n')[1:])
    else:
        return f'{yaml_language_server_line}\n{content}'


def _load_existing_cases(
    cases_path: Path,
    test_cases_type: type[TestCases[Any, Any, Any]],
) -> list[TestCase[Any, Any, Any]]:
    cases_text = cases_path.read_text()
    try:
        loaded = yaml.safe_load(cases_text)
    except yaml.YAMLError:
        raise ValueError(f'Cases file {cases_path} is not valid YAML')

    try:
        return test_cases_type.model_validate(loaded).test_cases
    except ValidationError as e:
        raise ValueError(
            f'Cases file {cases_path} contains data that does not match the schema. Delete the file before calling this function.'
        ) from e


async def _generate_examples(
    test_cases_type: type[TestCases[Any, Any, Any]],
    cases_path: Path,
    generate_examples_model: models.Model | models.KnownModelName = 'gpt-4o',
    n_examples: int = 3,
) -> list[TestCase[Any, Any, Any]]:
    if cases_path.exists():
        cases_text = cases_path.read_text()
        try:
            loaded = yaml.safe_load(cases_text)
        except yaml.YAMLError:
            raise ValueError(f'Cases file {cases_path} is not valid YAML')

        try:
            existing_cases = test_cases_type.model_validate(loaded).test_cases
        except ValidationError as e:
            raise ValueError(
                f'Cases file {cases_path} contains data that does not match the schema. Delete the file before calling this function.'
            ) from e
    else:
        existing_cases = []

    n_examples = max(0, n_examples - len(existing_cases))
    if n_examples == 0:
        return []

    agent = Agent(
        generate_examples_model,
        system_prompt=dedent('Generate concise example test cases that comply with the provided JSON schema.'),
        result_type=test_cases_type,
        retries=1,
    )
    return (await agent.run(f'Generate {n_examples} examples')).data.test_cases


# TODO: Use this elsewhere in the library to make it easier to reuse
# def infer_name(value: Any, function_frame: types.FrameType | None) -> str | None:
#     """Infer the variable name from the call frame.
#
#     Usage should be `infer_name(value, inspect.currentframe())`.
#     """
#     if function_frame is not None and (parent_frame := function_frame.f_back):  # pragma: no branch
#         for name, item in parent_frame.f_locals.items():
#             if item is value:
#                 return name
#         if parent_frame.f_locals != parent_frame.f_globals:
#             # if we couldn't find the agent in locals and globals are a different dict, try globals
#             for name, item in parent_frame.f_globals.items():
#                 if item is value:
#                     return name
#     return None
