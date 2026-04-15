from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

from pydantic import BaseModel, Field

from grasp_agent_middleware.agent import AgentController, AgentTool
from grasp_agent_middleware.backends import (
    SegmentationBackend,
    build_llm_backend,
    build_segmentation_backend,
)
from grasp_agent_middleware.config import Settings
from grasp_agent_middleware.execution import ExecutionAdapter
from grasp_agent_middleware.observability import ArtifactStore, event
from grasp_agent_middleware.perception import PerceptionService
from grasp_agent_middleware.planning import PlanningService
from grasp_agent_middleware.schemas import (
    AgentTraceStep,
    DryRunReport,
    ExecutionCommand,
    ObjectTable,
    PipelineRequest,
    PipelineResponse,
    Plan,
    PlanningRequest,
    SegmentationCandidate,
    ValidationStatus,
)


logger = logging.getLogger(__name__)


class WorkflowState(BaseModel):
    request_id: str
    request: PipelineRequest
    node_status: dict[str, str] = Field(default_factory=dict)
    segments: list[SegmentationCandidate] = Field(default_factory=list)
    object_table: ObjectTable | None = None
    plan: Plan | None = None
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)
    execution_commands: list[ExecutionCommand] = Field(default_factory=list)
    dry_run: DryRunReport | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    def mark(self, node: str, status: str) -> None:
        self.node_status[node] = status


@dataclass
class AgentWorkflow:
    segmentation_backend: SegmentationBackend
    perception: PerceptionService
    planning: PlanningService
    execution_adapter: ExecutionAdapter
    artifact_store: ArtifactStore
    agent_controller: AgentController
    enable_failure_dumps: bool = True

    def run(self, request: PipelineRequest) -> PipelineResponse:
        request_id = request.request_id or uuid4().hex
        request = request.model_copy(update={"request_id": request_id})
        state = WorkflowState(request_id=request_id, request=request)

        try:
            event(logger, request_id, "pipeline_started")
            self.agent_controller.run(state, self._agent_tools())
            state.artifacts["agent_trace"] = self.artifact_store.write_json(
                state.request_id, "agent_trace.json", state.agent_trace
            )
            state.artifacts["workflow_state"] = self.artifact_store.write_json(
                state.request_id, "workflow_state.json", state
            )
            event(logger, request_id, "pipeline_completed")
        except Exception as exc:
            state.errors.append(str(exc))
            if self.enable_failure_dumps:
                state.artifacts["failure"] = self.artifact_store.dump_failure(
                    request_id, exc, state.model_dump(mode="json")
                )
            event(logger, request_id, "pipeline_failed")
            raise

        assert state.object_table is not None
        assert state.plan is not None
        return PipelineResponse(
            request_id=request_id,
            object_table=state.object_table,
            plan=state.plan,
            agent_trace=state.agent_trace,
            execution_commands=state.execution_commands,
            dry_run=state.dry_run,
            artifacts=state.artifacts,
            validation=state.plan.validation,
        )

    def perceive(self, request: PipelineRequest) -> ObjectTable:
        request_id = request.request_id or uuid4().hex
        request = request.model_copy(update={"request_id": request_id})
        state = WorkflowState(request_id=request_id, request=request)
        self._segmentation_node(state)
        self._perception_node(state)
        self._perception_validation_node(state)
        assert state.object_table is not None
        state.artifacts["object_table"] = self.artifact_store.write_json(
            request_id, "object_table.json", state.object_table
        )
        return state.object_table

    def plan(self, request: PlanningRequest) -> Plan:
        plan = self.planning.create_plan(request.instruction, request.object_table)
        return self.planning.repair(plan, request.object_table)

    def build_execution_commands(
        self, object_table: ObjectTable, plan: Plan
    ) -> list[ExecutionCommand]:
        return self.execution_adapter.build_commands(object_table, plan)

    def dry_run_execution(self, object_table: ObjectTable, plan: Plan) -> DryRunReport:
        commands = self.build_execution_commands(object_table, plan)
        return self.execution_adapter.dry_run(
            request_id=plan.request_id,
            object_table=object_table,
            plan=plan,
            commands=commands,
        )

    def _agent_tools(self) -> list[AgentTool]:
        return [
            AgentTool(
                name="segment_tool",
                description="Run segmentation before structured perception.",
                should_run=lambda state: not state.segments,
                run=self._segmentation_node,
                summarize_input=lambda state: {
                    "image_ref": state.request.image.display_ref(),
                    "instruction": state.request.instruction,
                },
                summarize_output=lambda state: {"segments": len(state.segments)},
            ),
            AgentTool(
                name="perceive_tool",
                description="Fuse visual candidates into a schema-validated ObjectTable.",
                should_run=lambda state: bool(state.segments) and state.object_table is None,
                run=self._perception_node,
                summarize_input=lambda state: {"segments": len(state.segments)},
                summarize_output=lambda state: {
                    "objects": len(state.object_table.objects) if state.object_table else 0,
                    "uncertain": len(state.object_table.uncertain_objects) if state.object_table else 0,
                },
            ),
            AgentTool(
                name="validate_objects_tool",
                description="Check that perception produced execution-addressable objects.",
                should_run=lambda state: state.object_table is not None,
                run=self._perception_validation_node,
                summarize_input=lambda state: {
                    "objects": len(state.object_table.objects) if state.object_table else 0
                },
                summarize_output=lambda state: {"errors": list(state.errors)},
            ),
            AgentTool(
                name="plan_tool",
                description="Generate rules, assignments, and executable pick/place steps.",
                should_run=lambda state: state.object_table is not None and state.plan is None,
                run=self._planning_node,
                summarize_input=lambda state: {
                    "objects": len(state.object_table.objects) if state.object_table else 0
                },
                summarize_output=lambda state: {
                    "assignments": len(state.plan.assignments) if state.plan else 0,
                    "steps": len(state.plan.steps) if state.plan else 0,
                },
            ),
            AgentTool(
                name="validate_plan_tool",
                description="Verify object references and plan structural validity.",
                should_run=lambda state: state.plan is not None,
                run=self._plan_validation_node,
                summarize_input=lambda state: {"steps": len(state.plan.steps) if state.plan else 0},
                summarize_output=lambda state: {
                    "status": state.plan.validation.status if state.plan else None,
                    "warnings": len(state.plan.validation.warnings) if state.plan else 0,
                    "errors": len(state.plan.validation.errors) if state.plan else 0,
                },
            ),
            AgentTool(
                name="repair_tool",
                description="Repair failed or incomplete plans before export.",
                should_run=lambda state: state.plan is not None
                and (
                    state.plan.validation.status == ValidationStatus.failed
                    or bool(state.plan.validation.warnings)
                ),
                run=self._repair_node,
                summarize_input=lambda state: {
                    "status": state.plan.validation.status if state.plan else None,
                    "warnings": len(state.plan.validation.warnings) if state.plan else 0,
                },
                summarize_output=lambda state: {
                    "status": state.plan.validation.status if state.plan else None,
                    "repairs": len(state.plan.validation.repaired) if state.plan else 0,
                },
            ),
            AgentTool(
                name="execution_adapter_tool",
                description="Convert the validated plan into AnyGrasp-style command contracts.",
                should_run=lambda state: state.object_table is not None
                and state.plan is not None
                and not state.execution_commands,
                run=self._execution_adapter_node,
                summarize_input=lambda state: {"steps": len(state.plan.steps) if state.plan else 0},
                summarize_output=lambda state: {
                    "commands": len(state.execution_commands),
                    "dry_run": state.dry_run.status if state.dry_run else None,
                },
            ),
            AgentTool(
                name="output_tool",
                description="Persist final artifacts for debugging and replay.",
                should_run=lambda state: state.object_table is not None and state.plan is not None,
                run=self._output_node,
                summarize_input=lambda state: {
                    "objects": len(state.object_table.objects) if state.object_table else 0,
                    "steps": len(state.plan.steps) if state.plan else 0,
                    "commands": len(state.execution_commands),
                },
                summarize_output=lambda state: {"artifacts": sorted(state.artifacts.keys())},
            ),
        ]

    def _segmentation_node(self, state: WorkflowState) -> None:
        state.mark("segmentation", "running")
        state.segments = self.segmentation_backend.segment(
            state.request.image, state.request.instruction, state.request_id
        )
        state.artifacts["segmentation"] = self.artifact_store.write_json(
            state.request_id, "segmentation.json", state.segments
        )
        state.mark("segmentation", "completed")

    def _perception_node(self, state: WorkflowState) -> None:
        state.mark("perception", "running")
        state.object_table = self.perception.build_object_table(
            image=state.request.image,
            instruction=state.request.instruction,
            request_id=state.request_id,
            segments=state.segments,
        )
        state.mark("perception", "completed")

    def _perception_validation_node(self, state: WorkflowState) -> None:
        state.mark("perception_validation", "running")
        if state.object_table is None:
            raise RuntimeError("Perception node did not produce an ObjectTable.")
        if not state.object_table.objects:
            state.errors.append("ObjectTable is empty.")
        state.mark("perception_validation", "completed")

    def _planning_node(self, state: WorkflowState) -> None:
        state.mark("planning", "running")
        assert state.object_table is not None
        state.plan = self.planning.create_plan(state.request.instruction, state.object_table)
        state.mark("planning", "completed")

    def _plan_validation_node(self, state: WorkflowState) -> None:
        state.mark("plan_validation", "running")
        if state.plan is None:
            raise RuntimeError("Planning node did not produce a Plan.")
        if state.plan.validation.status == ValidationStatus.failed:
            state.errors.extend(state.plan.validation.errors)
        state.mark("plan_validation", "completed")

    def _repair_node(self, state: WorkflowState) -> None:
        state.mark("repair", "running")
        assert state.object_table is not None
        assert state.plan is not None
        state.plan = self.planning.repair(state.plan, state.object_table)
        state.mark("repair", "completed")

    def _execution_adapter_node(self, state: WorkflowState) -> None:
        state.mark("execution_adapter", "running")
        assert state.object_table is not None
        assert state.plan is not None
        state.execution_commands = self.execution_adapter.build_commands(
            state.object_table, state.plan
        )
        state.dry_run = self.execution_adapter.dry_run(
            request_id=state.request_id,
            object_table=state.object_table,
            plan=state.plan,
            commands=state.execution_commands,
        )
        state.mark("execution_adapter", "completed")

    def _output_node(self, state: WorkflowState) -> None:
        state.mark("output", "running")
        assert state.object_table is not None
        assert state.plan is not None
        state.artifacts["object_table"] = self.artifact_store.write_json(
            state.request_id, "object_table.json", state.object_table
        )
        state.artifacts["plan"] = self.artifact_store.write_json(
            state.request_id, "plan.json", state.plan
        )
        state.artifacts["execution_commands"] = self.artifact_store.write_json(
            state.request_id, "anygrasp_commands.json", state.execution_commands
        )
        if state.dry_run is not None:
            state.artifacts["dry_run"] = self.artifact_store.write_json(
                state.request_id, "dry_run.json", state.dry_run
            )
        state.mark("output", "completed")


def build_workflow(settings: Settings) -> AgentWorkflow:
    llm_backend = build_llm_backend(settings)
    segmentation_backend = build_segmentation_backend(settings)
    return AgentWorkflow(
        segmentation_backend=segmentation_backend,
        perception=PerceptionService(llm_backend=llm_backend),
        planning=PlanningService(),
        execution_adapter=ExecutionAdapter(),
        artifact_store=ArtifactStore(settings.artifact_dir),
        agent_controller=AgentController(),
        enable_failure_dumps=settings.enable_failure_dumps,
    )
