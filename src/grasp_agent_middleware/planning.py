from __future__ import annotations

import re
from dataclasses import dataclass

from grasp_agent_middleware.schemas import (
    Assignment,
    ObjectInstance,
    ObjectTable,
    Plan,
    PlanAction,
    PlanStep,
    Rule,
    RuleKind,
    ValidationReport,
    ValidationStatus,
)


_COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "white",
    "black",
    "orange",
    "purple",
    "gray",
    "grey",
]


def _normalize_target(target: str) -> str:
    target = target.strip().lower()
    target = re.sub(r"[^a-z0-9]+", "_", target)
    return target.strip("_") or "default_bin"


def _target_after(text: str, anchor: str) -> str | None:
    pattern = rf"\b{re.escape(anchor)}\b[^.;,\n]*?(?:to|into|in)\s+(.+?)(?=\s+(?:and|then)\b|[.;,\n]|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return _normalize_target(match.group(1))
    return None


@dataclass
class PlanningService:
    default_target: str = "default_bin"

    def parse_rules(self, instruction: str) -> list[Rule]:
        rules: list[Rule] = []
        text = instruction.lower()

        for color in _COLORS:
            if color in text:
                target = _target_after(text, color) or self.default_target
                rules.append(
                    Rule(
                        rule_id=f"rule_{len(rules) + 1:03d}",
                        kind=RuleKind.assignment,
                        description=f"Assign {color} objects to {target}.",
                        conditions={"attributes.color": color},
                        action={"target": target},
                        priority=100,
                    )
                )

        fragile_words = {"fragile", "glass", "delicate"}
        if any(word in instruction.lower() for word in fragile_words):
            rules.append(
                Rule(
                    rule_id=f"rule_{len(rules) + 1:03d}",
                    kind=RuleKind.safety,
                    description="Use conservative grasping for fragile objects.",
                    conditions={"attributes.material": ["glass"], "attributes.state": ["fragile"]},
                    action={"grasp_speed": "slow", "force_limit": "low"},
                    priority=10,
                )
            )

        if not rules:
            rules.append(
                Rule(
                    rule_id="rule_001",
                    kind=RuleKind.assignment,
                    description=f"Assign all sortable objects to {self.default_target}.",
                    conditions={"graspable": True},
                    action={"target": self.default_target},
                    priority=500,
                )
            )

        return rules

    def generate_assignments(
        self, object_table: ObjectTable, rules: list[Rule]
    ) -> list[Assignment]:
        assignments: list[Assignment] = []
        assignment_rules = [rule for rule in rules if rule.kind == RuleKind.assignment]

        for obj in object_table.objects:
            rule = self._match_rule(obj, assignment_rules)
            if rule is None:
                continue
            assignments.append(
                Assignment(
                    assignment_id=f"assign_{len(assignments) + 1:03d}",
                    object_id=obj.object_id,
                    target=str(rule.action.get("target", self.default_target)),
                    rule_id=rule.rule_id,
                    rationale=f"Matched {obj.object_id} against {rule.rule_id}.",
                    constraints=["verify_pose_before_pick"] if obj.object_id in object_table.uncertain_objects else [],
                    confidence=max(obj.confidence, 0.25),
                )
            )

        return assignments

    def generate_steps(self, assignments: list[Assignment], object_table: ObjectTable) -> list[PlanStep]:
        steps: list[PlanStep] = []
        uncertain = set(object_table.uncertain_objects)

        for assignment in assignments:
            if assignment.object_id in uncertain:
                steps.append(
                    PlanStep(
                        step_id=len(steps) + 1,
                        action=PlanAction.inspect,
                        object_id=assignment.object_id,
                        arguments={"reason": "low_confidence_perception"},
                        expected_result="Object identity and pose are verified.",
                    )
                )
            steps.append(
                PlanStep(
                    step_id=len(steps) + 1,
                    action=PlanAction.pick,
                    object_id=assignment.object_id,
                    arguments={"planner": "AnyGrasp", "reference": assignment.object_id},
                    preconditions=["object_pose_available", "gripper_empty"],
                    expected_result="Object is grasped.",
                )
            )
            steps.append(
                PlanStep(
                    step_id=len(steps) + 1,
                    action=PlanAction.place,
                    object_id=assignment.object_id,
                    target=assignment.target,
                    arguments={"target": assignment.target},
                    preconditions=["object_grasped"],
                    expected_result=f"Object is placed in {assignment.target}.",
                )
            )

        return steps

    def create_plan(self, instruction: str, object_table: ObjectTable) -> Plan:
        rules = self.parse_rules(instruction)
        assignments = self.generate_assignments(object_table, rules)
        steps = self.generate_steps(assignments, object_table)
        validation = self.validate(object_table=object_table, assignments=assignments, steps=steps)

        return Plan(
            request_id=object_table.request_id,
            instruction=instruction,
            object_table_ref=f"artifact://{object_table.request_id}/object_table.json",
            rules=rules,
            assignments=assignments,
            steps=steps,
            validation=validation,
            execution_contract={
                "consumer": "AnyGrasp-compatible execution layer",
                "object_reference_key": "object_id",
                "pose_source": "execution_ref",
                "step_arguments_are_structured": True,
            },
            metadata={"planner": "rule_based_execution_ready_planner"},
        )

    def validate(
        self,
        *,
        object_table: ObjectTable,
        assignments: list[Assignment],
        steps: list[PlanStep],
    ) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        object_ids = {obj.object_id for obj in object_table.objects}
        assigned_ids = {assignment.object_id for assignment in assignments}

        for assignment in assignments:
            if assignment.object_id not in object_ids:
                errors.append(f"Assignment references missing object_id={assignment.object_id}.")
        for step in steps:
            if step.object_id and step.object_id not in object_ids:
                errors.append(f"Plan step {step.step_id} references missing object_id={step.object_id}.")

        unassigned = sorted(object_ids - assigned_ids)
        if unassigned:
            warnings.append(f"Unassigned objects require inspect/skip decision: {', '.join(unassigned)}.")
        if not steps and object_ids:
            warnings.append("No executable steps were generated for detected objects.")
        if not object_ids:
            errors.append("No objects available for planning.")

        return ValidationReport(
            status=ValidationStatus.failed if errors else ValidationStatus.passed,
            errors=errors,
            warnings=warnings,
        )

    def repair(self, plan: Plan, object_table: ObjectTable) -> Plan:
        if plan.validation.status == ValidationStatus.passed and not plan.validation.warnings:
            return plan

        repaired = list(plan.validation.repaired)
        steps = list(plan.steps)
        assigned_ids = {assignment.object_id for assignment in plan.assignments}

        for obj in object_table.objects:
            if obj.object_id in assigned_ids:
                continue
            steps.append(
                PlanStep(
                    step_id=len(steps) + 1,
                    action=PlanAction.inspect,
                    object_id=obj.object_id,
                    arguments={"reason": "unassigned_object"},
                    expected_result="Operator or downstream policy decides whether to skip or assign.",
                )
            )
            steps.append(
                PlanStep(
                    step_id=len(steps) + 1,
                    action=PlanAction.skip,
                    object_id=obj.object_id,
                    arguments={"reason": "no_matching_assignment_rule"},
                    expected_result="Object is left untouched.",
                )
            )
            repaired.append(f"Added inspect/skip fallback for {obj.object_id}.")

        validation = self.validate(
            object_table=object_table,
            assignments=plan.assignments,
            steps=steps,
        )
        status = ValidationStatus.repaired if not validation.errors else ValidationStatus.failed
        validation = validation.model_copy(update={"status": status, "repaired": repaired})
        return plan.model_copy(update={"steps": steps, "validation": validation})

    def _match_rule(self, obj: ObjectInstance, rules: list[Rule]) -> Rule | None:
        for rule in sorted(rules, key=lambda item: item.priority):
            conditions = rule.conditions
            color = conditions.get("attributes.color")
            if color and obj.attributes.color == color:
                return rule
            if conditions.get("graspable") is True and obj.graspable:
                return rule
        return None
