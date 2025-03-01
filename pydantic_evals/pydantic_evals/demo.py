from functools import partial

from pydantic import AwareDatetime, BaseModel
from typing_extensions import TypedDict

from pydantic_evals import evaluation
from pydantic_evals.demo_agent import (
    TimeRangeAgentResponse,
    infer_time_range,
)
from pydantic_evals.evals import EvalCase
from pydantic_evals.llm_as_a_judge import GradingOutput, judge_input_output


class TimeRangeInputs(TypedDict):
    """The inputs for a time range inference agent."""

    prompt: str
    now: AwareDatetime


class TimeRangeExample(BaseModel):
    """An example of a time range inference agent input and output."""

    name: str
    inputs: TimeRangeInputs
    expected_output: TimeRangeAgentResponse


raw_cases = [
    {
        'name': 'yesterday',
        'inputs': {'prompt': 'yesterday from 2-4 ET', 'now': '2024-01-01T00:00:00Z'},
        'expected_output': {
            'min_timestamp_with_offset': '2024-01-01T00:00:00Z',
            'max_timestamp_with_offset': '2024-01-01T00:00:00Z',
            'explanation': '...',
        },
    },
    {
        'name': 'last 24 hours',
        'inputs': {'prompt': 'the last 24 hours', 'now': '2024-01-01T00:00:00Z'},
        'expected_output': {
            'min_timestamp_with_offset': '2024-01-01T00:00:00Z',
            'max_timestamp_with_offset': '2024-01-01T00:00:00Z',
            'explanation': '...',
        },
    },
    {
        'name': 'specific time range',
        'inputs': {'prompt': '6 to 9 PM ET on October 8th', 'now': '2024-01-01T00:00:00Z'},
        'expected_output': {
            'min_timestamp_with_offset': '2024-01-01T00:00:00Z',
            'max_timestamp_with_offset': '2024-01-01T00:00:00Z',
            'explanation': '...',
        },
    },
    {
        'name': 'next week',
        'inputs': {'prompt': 'next week', 'now': '2024-01-01T00:00:00Z'},
        'expected_output': {
            'min_timestamp_with_offset': '2024-01-01T00:00:00Z',
            'max_timestamp_with_offset': '2024-01-01T00:00:00Z',
            'explanation': '...',
        },
    },
    {
        'name': 'invalid - question',
        'inputs': {'prompt': 'what time is it?', 'now': '2024-01-01T00:00:00Z'},
        'expected_output': {
            'min_timestamp_with_offset': '2024-01-01T00:00:00Z',
            'max_timestamp_with_offset': '2024-01-01T00:00:00Z',
            'explanation': '...',
        },
    },
]
cases = [TimeRangeExample.model_validate(raw_case) for raw_case in raw_cases]


async def judge_time_range_case(inputs: TimeRangeInputs, output: TimeRangeAgentResponse) -> GradingOutput:
    """Judge the output of a time range inference agent based on a rubric."""
    rubric = 'The output should be a reasonable time range to select for the given inputs.'
    return await judge_input_output(inputs, output, rubric)


async def main():
    """TODO: Remove this file before merging."""
    import logfire

    logfire.configure(send_to_logfire=False, console=logfire.ConsoleOptions(verbose=True))

    async def handle_case(eval_case: EvalCase[..., TimeRangeAgentResponse], inputs: TimeRangeInputs):
        result = await judge_time_range_case(inputs=inputs, output=eval_case.output)
        eval_case.record_label('reasonable', 'yes' if result else 'no')

    async with evaluation(infer_time_range) as my_eval:
        for case_data in cases:
            bound_handler = partial(handle_case, inputs=case_data.inputs)
            my_eval.case(name=case_data.name).call(**case_data.inputs).parallel_handler(bound_handler)

    my_eval.print_report(include_input=True, include_output=True)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
