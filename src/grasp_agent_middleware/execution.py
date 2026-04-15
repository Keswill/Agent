from __future__ import annotations

from dataclasses import dataclass

from grasp_agent_middleware.schemas import (
    DryRunEvent,
    DryRunReport,
    ExecutionCommand,
    ObjectInstance,
    ObjectTable,
    Plan,
    PlanAction,
    ValidationStatus,
)


@dataclass
class ExecutionAdapter:
    """Converts validated plan steps into robot-side command contracts."""

    grasp_planner: str = "AnyGrasp"

    def build_commands(self, object_table: ObjectTable, plan: Plan) -> list[ExecutionCommand]:
        objects = {obj.object_id: obj for obj in object_table.objects}
        commands: list[ExecutionCommand] = []

        for step in plan.steps:
            obj = objects.get(step.object_id or "")
            command = ExecutionCommand(
                command_id=f"cmd_{len(commands) + 1:03d}",
                step_id=step.step_id,
                action=step.action,
                object_id=step.object_id,
                target=step.target,
                frame_id=self._frame_id(obj),
                object_ref=self._object_ref(obj),
                bbox=obj.bbox if obj else None,
                mask=obj.mask if obj else None,
                grasp_planner=self.grasp_planner if step.action == PlanAction.pick else None,
                parameters=self._parameters(step.arguments, obj),
            )
            commands.append(command)

        return commands

    def dry_run(
        self,
        *,
        request_id: str,
        object_table: ObjectTable,
        plan: Plan,
        commands: list[ExecutionCommand],
    ) -> DryRunReport:
        object_ids = {obj.object_id for obj in object_table.objects}
        errors: list[str] = []
        events: list[DryRunEvent] = []

        for command in commands:
            if command.action in {PlanAction.pick, PlanAction.inspect} and not command.object_id:
                errors.append(f"{command.command_id} is missing object_id.")
                events.append(
                    DryRunEvent(
                        command_id=command.command_id,
                        status="blocked",
                        message="Object reference is required for this command.",
                    )
                )
                continue
            if command.object_id and command.object_id not in object_ids:
                errors.append(f"{command.command_id} references missing object_id={command.object_id}.")
                events.append(
                    DryRunEvent(
                        command_id=command.command_id,
                        status="blocked",
                        message="Object reference is not present in the object table.",
                    )
                )
                continue
            if command.action == PlanAction.place and not command.target:
                errors.append(f"{command.command_id} place command is missing target.")
                events.append(
                    DryRunEvent(
                        command_id=command.command_id,
                        status="blocked",
                        message="Place target is required.",
                    )
                )
                continue
            events.append(
                DryRunEvent(
                    command_id=command.command_id,
                    status="ready" if command.action != PlanAction.skip else "skipped",
                    message=f"{command.action.value} command is structurally executable.",
                )
            )

        return DryRunReport(
            request_id=request_id,
            status=ValidationStatus.failed if errors else ValidationStatus.passed,
            commands_checked=len(commands),
            events=events,
            errors=errors,
        )

    def _parameters(self, step_args: dict, obj: ObjectInstance | None) -> dict:
        parameters = dict(step_args)
        if obj and obj.execution_ref:
            parameters.setdefault("execution_ref", obj.execution_ref.model_dump(mode="json"))
        return parameters

    def _frame_id(self, obj: ObjectInstance | None) -> str:
        if obj and obj.execution_ref:
            return obj.execution_ref.frame_id
        return "camera"

    def _object_ref(self, obj: ObjectInstance | None) -> str | None:
        if obj and obj.execution_ref:
            return obj.execution_ref.object_ref
        return obj.object_id if obj else None

