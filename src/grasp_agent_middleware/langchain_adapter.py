from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from grasp_agent_middleware.orchestration import AgentWorkflow
from grasp_agent_middleware.schemas import ExecutionRequest, PerceptionRequest, PipelineRequest, PlanningRequest


class ToolManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    integration: str = "langchain"
    tools: list[dict[str, str]]
    install_hint: str | None = None


@dataclass
class LangChainToolAdapter:
    """Optional LangChain-compatible wrapper around the native workflow tools."""

    workflow: AgentWorkflow

    def manifest(self) -> ToolManifest:
        available = _langchain_available()
        return ToolManifest(
            available=available,
            tools=[
                {
                    "name": "robotic_sorting_pipeline",
                    "description": "Run the full image + instruction to ObjectTable, Plan, trace, and commands pipeline.",
                },
                {
                    "name": "robotic_perceive",
                    "description": "Run segmentation and perception only; returns ObjectTable.",
                },
                {
                    "name": "robotic_plan",
                    "description": "Generate and repair Plan from instruction and ObjectTable.",
                },
                {
                    "name": "robotic_execution_commands",
                    "description": "Convert ObjectTable + Plan into AnyGrasp-style ExecutionCommand objects.",
                },
                {
                    "name": "robotic_execution_dry_run",
                    "description": "Validate execution command structure before robot handoff.",
                },
            ],
            install_hint=None if available else 'Install optional deps with: python -m pip install -e ".[agent]"',
        )

    def build_tools(self) -> list[Any]:
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as exc:
            raise RuntimeError(
                'LangChain tools require optional deps. Install with: python -m pip install -e ".[agent]"'
            ) from exc

        return [
            StructuredTool.from_function(
                name="robotic_sorting_pipeline",
                description="Run the full robotic sorting middleware pipeline.",
                args_schema=PipelineRequest,
                func=self._pipeline_tool,
            ),
            StructuredTool.from_function(
                name="robotic_perceive",
                description="Run segmentation and perception to build an ObjectTable.",
                args_schema=PerceptionRequest,
                func=self._perceive_tool,
            ),
            StructuredTool.from_function(
                name="robotic_plan",
                description="Generate and repair an execution-ready Plan.",
                args_schema=PlanningRequest,
                func=self._plan_tool,
            ),
            StructuredTool.from_function(
                name="robotic_execution_commands",
                description="Build AnyGrasp-style commands from ObjectTable and Plan.",
                args_schema=ExecutionRequest,
                func=self._execution_commands_tool,
            ),
            StructuredTool.from_function(
                name="robotic_execution_dry_run",
                description="Run structural dry-run checks for execution commands.",
                args_schema=ExecutionRequest,
                func=self._dry_run_tool,
            ),
        ]

    def _pipeline_tool(self, **kwargs: Any) -> dict[str, Any]:
        request = PipelineRequest.model_validate(kwargs)
        return self.workflow.run(request).model_dump(mode="json")

    def _perceive_tool(self, **kwargs: Any) -> dict[str, Any]:
        request = PerceptionRequest.model_validate(kwargs)
        pipeline_request = PipelineRequest(
            image=request.image,
            instruction=request.instruction,
            request_id=request.request_id,
        )
        return self.workflow.perceive(pipeline_request).model_dump(mode="json")

    def _plan_tool(self, **kwargs: Any) -> dict[str, Any]:
        request = PlanningRequest.model_validate(kwargs)
        return self.workflow.plan(request).model_dump(mode="json")

    def _execution_commands_tool(self, **kwargs: Any) -> list[dict[str, Any]]:
        request = ExecutionRequest.model_validate(kwargs)
        return [
            command.model_dump(mode="json")
            for command in self.workflow.build_execution_commands(request.object_table, request.plan)
        ]

    def _dry_run_tool(self, **kwargs: Any) -> dict[str, Any]:
        request = ExecutionRequest.model_validate(kwargs)
        return self.workflow.dry_run_execution(request.object_table, request.plan).model_dump(mode="json")


def _langchain_available() -> bool:
    try:
        import langchain_core.tools  # noqa: F401
    except ImportError:
        return False
    return True

