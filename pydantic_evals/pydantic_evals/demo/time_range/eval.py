from functools import partial

from pydantic import AwareDatetime

from pydantic_evals import evaluation
from pydantic_evals.demo.time_range import TimeRangeAgentResponse, infer_time_range
from pydantic_evals.demo.time_range.models import TimeRangeInputs, TimeRangeTestCases
from pydantic_evals.evals import EvalCase
from pydantic_evals.llm_as_a_judge import GradingOutput, judge_input_output


async def judge_time_range_case(inputs: TimeRangeInputs, output: TimeRangeAgentResponse) -> GradingOutput:
    """Judge the output of a time range inference agent based on a rubric."""
    rubric = 'The output should be a reasonable time range to select for the given inputs.'
    return await judge_input_output(inputs, output, rubric)


async def main():
    """TODO: Remove this file before merging."""
    import logfire

    logfire.configure(send_to_logfire=False, console=logfire.ConsoleOptions(verbose=True))

    async def handle_case(
        eval_case: EvalCase[[str, AwareDatetime | None], TimeRangeAgentResponse], inputs: TimeRangeInputs
    ):
        result = await judge_time_range_case(inputs=inputs, output=eval_case.output)
        eval_case.record_label('reasonable', 'yes' if result else 'no')

    cases = TimeRangeTestCases.from_yaml().test_cases
    async with evaluation(infer_time_range) as my_eval:
        for case_data in cases:
            bound_handler = partial(handle_case, inputs=case_data.inputs)
            my_eval.case(name=case_data.name).call(**case_data.inputs).parallel_handler(bound_handler)

    my_eval.print_report(include_input=True, include_output=True)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
