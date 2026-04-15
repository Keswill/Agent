from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from grasp_agent_middleware.schemas import AgentToolStatus, AgentTraceStep, utc_now


StatePredicate = Callable[[Any], bool]
ToolRunner = Callable[[Any], None]
SummaryBuilder = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    should_run: StatePredicate
    run: ToolRunner
    summarize_input: SummaryBuilder
    summarize_output: SummaryBuilder


@dataclass
class AgentController:
    """Small state-machine agent that selects tools and records a replayable trace."""

    max_steps: int = 12

    def run(self, state: Any, tools: list[AgentTool]) -> None:
        for tool in tools[: self.max_steps]:
            trace_id = len(state.agent_trace) + 1
            should_run = tool.should_run(state)
            trace = AgentTraceStep(
                trace_id=trace_id,
                tool_name=tool.name,
                decision=tool.description if should_run else f"Skip {tool.name}; prerequisites not met.",
                status=AgentToolStatus.selected if should_run else AgentToolStatus.skipped,
                input_summary=tool.summarize_input(state),
            )
            if not should_run:
                trace.ended_at = utc_now()
                trace.latency_ms = 0.0
                state.agent_trace.append(trace)
                continue

            start = perf_counter()
            try:
                tool.run(state)
                trace.status = AgentToolStatus.completed
                trace.output_summary = tool.summarize_output(state)
            except Exception as exc:
                trace.status = AgentToolStatus.failed
                trace.errors.append(str(exc))
                raise
            finally:
                trace.ended_at = utc_now()
                trace.latency_ms = round((perf_counter() - start) * 1000, 3)
                state.agent_trace.append(trace)

