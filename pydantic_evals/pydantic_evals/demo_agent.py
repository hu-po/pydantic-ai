from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import logfire
from logfire import ConsoleOptions
from pydantic import AwareDatetime, BaseModel

from pydantic_ai import Agent, RunContext


class TimeRangeBuilderSuccess(BaseModel, use_attribute_docstrings=True):
    """Response when a time range could be successfully generated."""

    min_timestamp_with_offset: AwareDatetime
    """A datetime in ISO format with timezone offset."""

    max_timestamp_with_offset: AwareDatetime
    """A datetime in ISO format with timezone offset."""

    explanation: str | None
    """
    A brief explanation of the time range that was selected.

    For example, if a user only mentions a specific point in time, you might explain that you selected a 10 minute
    window around that time.
    """

    def __str__(self):
        readable_min_timestamp = self.min_timestamp_with_offset.strftime('%A, %B %d, %Y %H:%M:%S %Z')
        readable_max_timestamp = self.max_timestamp_with_offset.strftime('%A, %B %d, %Y %H:%M:%S %Z')
        lines = [
            'TimeRangeBuilderSuccess:',
            f'* min_timestamp_with_offset: {readable_min_timestamp}',
            f'* max_timestamp_with_offset: {readable_max_timestamp}',
        ]
        if self.explanation is not None:
            lines.append(f'* explanation: {self.explanation}')
        return '\n'.join(lines)


class TimeRangeBuilderError(BaseModel):
    """Response when a time range cannot not be generated."""

    error_message: str

    def __str__(self):
        return f'TimeRangeBuilderError:\n* {self.error_message}'


TimeRangeAgentResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError


@dataclass
class TimeRangeDeps:
    """Dependencies for the time range inference agent."""

    now: datetime = field(default_factory=lambda: datetime.now().astimezone())


def time_range_system_prompt(ctx: RunContext[TimeRangeDeps]):
    """Build the system prompt for the time range inference agent."""
    # Format like: Friday, November 22, 2024 11:15:14 PST
    now_str = ctx.deps.now.strftime('%A, %B %d, %Y %H:%M:%S %Z')
    return f"""\
Convert the user's request into a structured time range. Both the min and max must have a timezone offset specified.
If the user does not request a specific timezone, use the offset from their local time (provided below).
If ambiguous, prefer to select a time range in the (recent) past.

If the user's request is too ambiguous or cannot be converted into a time range, return an error message.
If the user mentions a specific point in time, select a 10 minute window around that time.

In the explanation field, include a brief message addressed to the user in passive voice indicating how the time range has been updated.
If you had to interpret anything possibly-ambiguous in the user's request, please address that in your response.
In the explanation, **DO NOT** repeat anything about the user's request that was unambiguous.

Examples:
- If the user says "yesterday", you should say "Selected midnight to midnight yesterday."
- If the user says "the last 24 hours", you should say "The time range has been updated."
- If the user says "next week", you might say "Selected a 7-day window starting now."
- If the user says "around noon Tokyo time" you might say "Selected a 10-minute window around 12PM JST."

The user's local time is {now_str}.
"""


time_range_agent = Agent[TimeRangeDeps, TimeRangeAgentResponse](
    'gpt-4o',
    # we can't yet annotate type form, hence type ignore
    result_type=TimeRangeAgentResponse,  # type: ignore
    deps_type=TimeRangeDeps,
    retries=1,
)
time_range_agent.system_prompt(time_range_system_prompt)


async def infer_time_range(prompt: str, now: AwareDatetime | None = None) -> TimeRangeAgentResponse:
    """Infer a time range from a user prompt."""
    deps = TimeRangeDeps(now=now or datetime.now().astimezone())
    return (await time_range_agent.run(prompt, deps=deps)).data


if __name__ == '__main__':

    async def main():
        """Example usage of the time range inference agent."""
        logfire.configure(send_to_logfire=False, console=ConsoleOptions(verbose=True))
        user_prompt = 'yesterday from 2-4 ET'
        # user_prompt = 'the last 24 hours'
        # user_prompt = '6 to 9 PM ET on October 8th'
        # user_prompt = 'next week'
        # user_prompt = 'what time is it?'

        print(await infer_time_range(user_prompt))

    asyncio.run(main())
