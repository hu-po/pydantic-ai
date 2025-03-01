from __future__ import annotations as _annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Generic, ParamSpec, TypeVar

import logfire_api
from rich.console import Console

from pydantic_evals._utils import UNSET, Unset
from pydantic_evals.reports import EvalReport, EvalReportCase, RenderNumberConfig, RenderValueConfig

__all__ = ('Eval', 'EvalCase', 'evaluation', 'increment_eval_metric')

P = ParamSpec('P')
OutputT = TypeVar('OutputT')


@asynccontextmanager
async def evaluation(
    task: Callable[P, Awaitable[OutputT]], *, name: str | None = None
) -> AsyncIterator[Eval[P, OutputT]]:
    """Context manager for starting an evaluation."""
    with logfire_api.span('evaluation', name=name) as eval_span:
        eval = Eval(task, name=name, span=eval_span)

        yield eval

        async def _handle_case(case: EvalCase[P, OutputT]) -> None:
            if not case.has_output:
                async with case:
                    pass

        async with asyncio.TaskGroup() as group:
            for eval_case in eval.cases:
                # TODO: Use a semaphore to enforce some kind of max concurrency.
                #  See https://discuss.python.org/t/adding-a-limit-parameter-to-taskgroup-for-concurrent-coroutines-management/47686/3
                group.create_task(_handle_case(eval_case), name=eval_case.name)


@dataclass
class Eval(Generic[P, OutputT]):
    """A container for evaluation cases.

    This should generally not be instantiated directly; instead, use the `evaluation` context manager.
    """

    task: Callable[P, Awaitable[OutputT]]
    task_name: str
    handler: Callable[[EvalCase[P, OutputT]], Awaitable[None]]
    span: logfire_api.LogfireSpan | None = field(repr=False)
    cases: list[EvalCase[..., Any]] = field(repr=False, default_factory=list)  # TODO: Should be list[EvalCase[OutputT]]

    def __init__(
        self,
        task: Callable[P, Awaitable[OutputT]],
        *,
        name: str | None = None,
        span: logfire_api.LogfireSpan | None = None,
    ):
        if name is None:
            name = _get_task_name(task)
        self.task = task

        self.name = name
        self.span = span
        self.cases: list[EvalCase[..., Any]] = []

    def as_report(self) -> EvalReport:
        return EvalReport(name=self.name, cases=[c.as_report_case() for c in self.cases])

    def print_report(
        self,
        width: int | None = None,
        include_input: bool = False,
        include_output: bool = False,
        include_total_duration: bool = False,
        include_averages: bool = True,
        input_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
    ) -> None:
        console = Console(width=width)
        console.print(
            self.as_report().console_table(
                include_input=include_input,
                include_output=include_output,
                include_total_duration=include_total_duration,
                include_averages=include_averages,
                input_config=input_config,
                output_config=output_config,
                score_configs=score_configs,
                label_configs=label_configs,
                metric_configs=metric_configs,
                duration_config=duration_config,
            )
        )

    def print_diff(
        self,
        *,
        baseline: Eval[P, OutputT] | EvalReport,
        width: int | None = None,
        include_input: bool = False,
        include_output: bool = False,
        include_total_duration: bool = False,
        include_removed_cases: bool = False,
        include_averages: bool = True,
        input_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
    ) -> None:
        if not isinstance(baseline, EvalReport):
            baseline = baseline.as_report()

        console = Console(width=width)
        console.print(
            self.as_report().console_table(
                baseline=baseline,
                include_input=include_input,
                include_output=include_output,
                include_total_duration=include_total_duration,
                include_removed_cases=include_removed_cases,
                include_averages=include_averages,
                input_config=input_config,
                output_config=output_config,
                score_configs=score_configs,
                label_configs=label_configs,
                metric_configs=metric_configs,
                duration_config=duration_config,
            )
        )

    def case(self, name: str) -> EvalCase[P, OutputT]:
        case = EvalCase(name, self)
        self.cases.append(case)
        return case


@dataclass(init=False)
class EvalCase(Generic[P, OutputT]):
    """A container for an evaluation case."""

    name: str
    eval: Eval[P, OutputT]

    scores: dict[str, int | float]
    metrics: dict[str, int | float]
    labels: dict[str, bool | str]
    metadata: dict[str, Any]

    def __init__(self, name: str, eval: Eval[P, OutputT]):
        self.name = name
        self.eval = eval

        self.scores = {}
        self.metrics = {}
        self.labels = {}
        self.metadata = {}

        self._inputs: dict[str, Any] | None = None
        self._default_inputs: dict[str, Any] | None = None
        self._bound_arguments: inspect.BoundArguments | None = None
        self._output: OutputT | Unset = UNSET

        self._case_span: logfire_api.LogfireSpan | None = None
        self._task_span: logfire_api.LogfireSpan | None = None
        self._parallel_handler: Callable[[EvalCase[P, OutputT]], Awaitable[None]] | None = None
        self._token: Any = None

    @property
    def inputs(self) -> dict[str, Any]:
        if self._inputs is None:
            raise RuntimeError('You must call `EvalCase.call` before accessing the inputs.')
        return self._inputs

    @property
    def bound_arguments(self) -> inspect.BoundArguments:
        if self._bound_arguments is None:
            raise RuntimeError('You must call `EvalCase.call` before accessing the bound_arguments.')
        return self._bound_arguments

    @property
    def case_span(self) -> logfire_api.LogfireSpan:
        if self._case_span is None:
            raise RuntimeError('You must use `with` before accessing the case_span.')
        return self._case_span

    @property
    def task_span(self) -> logfire_api.LogfireSpan:
        if self._task_span is None:
            raise RuntimeError('You must use `with` before accessing the task_span.')
        return self._task_span

    def call(self, *case_input_args: P.args, **case_input_kwargs: P.kwargs) -> EvalCase[P, OutputT]:
        signature = inspect.signature(self.eval.task)
        bound_arguments = signature.bind(*case_input_args, **case_input_kwargs)
        self._inputs = dict(bound_arguments.arguments)
        bound_arguments.apply_defaults()
        self._default_inputs = {k: v for k, v in bound_arguments.arguments.items() if k not in self._inputs}
        self._bound_arguments = bound_arguments

        self._case_span = logfire_api.span(
            'case: {name}',
            name=self.name,
            task_input=self._inputs,
            task_defaults=self._default_inputs,
        )
        self._task_span = logfire_api.span('task execution')
        return self

    def parallel_handler(self, handler: Callable[[EvalCase[P, OutputT]], Awaitable[None]]) -> EvalCase[P, OutputT]:
        self._parallel_handler = handler
        return self

    @property
    def has_output(self) -> bool:
        return not isinstance(self._output, Unset)

    @property
    def output(self) -> OutputT:
        if isinstance(self._output, Unset):
            raise RuntimeError('case_output accessed before it was set.')
        return self._output

    @output.setter
    def output(self, case_output: OutputT) -> None:
        self._output = case_output

    def record_score(self, name: str, value: int | float) -> None:
        score_attribute = f'score.{name}'
        self.scores[name] = value

        # If we want to use span links to store scores we can do something like this:
        # with logfire.span('score {name=} {value=}', name=name, value=value, _links=[(self.span.context, None)]):
        #     pass

        # We need to support updating scores via span links, but I'm not sure if we should _only_ support that
        self.case_span.set_attribute(score_attribute, value)

    def record_label(self, name: str, value: bool | str) -> None:
        label_attribute = f'label.{name}'
        self.labels[name] = value

        # If we want to use span links to store labels we can do something like this:
        # with logfire.span('label {name=} {value=}', name=name, value=value, _links=[(self.span.context, None)]):
        #     pass

        # We need to support updating labels via span links, but I'm not sure if we should _only_ support that
        self.case_span.set_attribute(label_attribute, value)

    def record_metadata(self, name: str, value: bool | str) -> None:
        label_attribute = f'label.{name}'
        self.metadata[name] = value

        # If we want to use span links to store labels we can do something like this:
        # with logfire.span('label {name=} {value=}', name=name, value=value, _links=[(self.span.context, None)]):
        #     pass

        # We need to support updating labels via span links, but I'm not sure if we should _only_ support that
        self.case_span.set_attribute(label_attribute, value)

    def record_metric(self, name: str, value: int | float) -> None:
        metric_attribute = f'metric.{name}'
        self.metrics[name] = value

        # If we want to use span links to store metrics we can do something like this:
        # with logfire.span('metric {name=} {value=}', name=name, value=value, _links=[(self.span.context, None)]):
        #     pass

        # We need to support updating metrics via span links, but I'm not sure if we should _only_ support that
        self.case_span.set_attribute(metric_attribute, value)

    def increment_metric(self, name: str, amount: int | float) -> None:
        current_value = self.metrics.get(name, 0)
        incremented_value = current_value + amount
        self.record_metric(name, incremented_value)

    def as_report_case(self) -> EvalReportCase:
        return EvalReportCase(
            name=self.name,
            inputs=self.inputs,
            output=self.output,
            scores=self.scores,
            metrics=self.metrics,
            labels=self.labels,
            task_duration=_get_span_duration(self.task_span),
            total_duration=_get_span_duration(self.case_span),
        )

    async def __aenter__(self):
        if _CURRENT_EVAL_CASE.get() is not None:
            raise RuntimeError('An eval case has already been entered. Evaluation cases should not be nested')
        self._token = _CURRENT_EVAL_CASE.set(self)
        self.case_span.__enter__()
        with self.task_span:
            case_output = await self.eval.task(*self.bound_arguments.args, **self.bound_arguments.kwargs)
        self.case_span.set_attribute('case_output', case_output)
        self.output = case_output
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any
    ) -> None:
        if self._parallel_handler is not None:
            await self._parallel_handler(self)
        self.case_span.__exit__(exc_type, exc_value, traceback)
        _CURRENT_EVAL_CASE.reset(self._token)


_CURRENT_EVAL_CASE = ContextVar[EvalCase[..., Any] | None]('_CURRENT_EVAL_CASE', default=None)


def increment_eval_metric(name: str, amount: int | float) -> None:
    """Increment the named metric for the current evaluation case."""
    current_case = _CURRENT_EVAL_CASE.get()
    if current_case is not None:
        current_case.increment_metric(name, amount)


def _get_task_name(func: Callable[..., Any]) -> str:
    def _unwrap(f: Callable[..., Any]) -> Callable[..., Any]:
        # Unwraps f, also unwrapping partials, for the sake of getting f's name
        if isinstance(f, partial):
            return _unwrap(f.func)
        return inspect.unwrap(f)

    return _unwrap(func).__qualname__


def _get_span_duration(span: logfire_api.LogfireSpan) -> float:
    end_time = span.end_time
    start_time = span.start_time
    assert isinstance(start_time, int), 'span is not started'
    assert isinstance(end_time, int), 'span is not finished'
    return (end_time - start_time) / 1_000_000_000
