from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError  # noqa: E402

from grasp_agent_middleware.config import Settings  # noqa: E402
from grasp_agent_middleware.observability import configure_logging  # noqa: E402
from grasp_agent_middleware.orchestration import build_workflow  # noqa: E402
from grasp_agent_middleware.schemas import PipelineRequest, PipelineResponse, ValidationStatus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run contract and behavior evals for the pipeline.")
    parser.add_argument("--eval-set", default="samples/eval_set.jsonl")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    settings = Settings.from_env()
    configure_logging(settings.log_level)
    workflow = build_workflow(settings)
    results: list[dict[str, Any]] = []

    for line in Path(args.eval_set).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        started = perf_counter()
        error = None
        response = None
        try:
            response = workflow.run(PipelineRequest.model_validate(case["request"]))
            PipelineResponse.model_validate(response.model_dump(mode="json"))
        except (Exception, ValidationError) as exc:
            error = str(exc)
        latency_ms = round((perf_counter() - started) * 1000, 3)
        result = evaluate_case(case, response, latency_ms, error)
        results.append(result)
        print(json.dumps(result, ensure_ascii=True))

    summary = summarize(results)
    write_report(Path(args.report_dir), results, summary)
    print(json.dumps(summary, ensure_ascii=True))


def evaluate_case(
    case: dict[str, Any],
    response: PipelineResponse | None,
    latency_ms: float,
    error: str | None,
) -> dict[str, Any]:
    if response is None:
        return {
            "case_id": case["case_id"],
            "ok": False,
            "latency_ms": latency_ms,
            "schema_valid": False,
            "object_reference_valid": False,
            "assignment_accuracy": 0.0,
            "plan_valid": False,
            "repair_success": False,
            "command_valid": False,
            "agent_trace_valid": False,
            "artifact_complete": False,
            "dry_run_coverage": False,
            "error": error,
        }

    expect = case.get("expect", {})
    object_ids = {obj.object_id for obj in response.object_table.objects}
    plan_ids = {step.object_id for step in response.plan.steps if step.object_id}
    command_ids = {command.object_id for command in response.execution_commands if command.object_id}
    object_reference_valid = plan_ids.issubset(object_ids) and command_ids.issubset(object_ids)
    assignment_accuracy = _assignment_accuracy(response, expect)
    actions = {step.action.value for step in response.plan.steps}
    required_actions = set(expect.get("required_actions", []))
    plan_valid = (
        response.validation.status in {ValidationStatus.passed, ValidationStatus.repaired}
        and required_actions.issubset(actions)
        and len(object_ids) >= int(expect.get("min_objects", 1))
    )
    repair_success = response.validation.status != ValidationStatus.failed
    command_valid = bool(response.execution_commands) and (
        response.dry_run is not None and response.dry_run.status == ValidationStatus.passed
    )
    agent_trace_valid = _agent_trace_valid(response)
    artifact_complete = _artifact_complete(response)
    dry_run_coverage = (
        response.dry_run is not None
        and response.dry_run.commands_checked == len(response.execution_commands)
        and len(response.dry_run.events) == len(response.execution_commands)
    )
    min_trace_steps = int(expect.get("min_trace_steps", 6))
    ok = (
        object_reference_valid
        and plan_valid
        and command_valid
        and agent_trace_valid
        and artifact_complete
        and dry_run_coverage
        and len(response.agent_trace) >= min_trace_steps
        and assignment_accuracy >= float(expect.get("min_assignment_accuracy", 0.0))
    )

    return {
        "case_id": case["case_id"],
        "ok": ok,
        "request_id": response.request_id,
        "latency_ms": latency_ms,
        "schema_valid": True,
        "objects": len(object_ids),
        "steps": len(response.plan.steps),
        "commands": len(response.execution_commands),
        "object_reference_valid": object_reference_valid,
        "assignment_accuracy": round(assignment_accuracy, 3),
        "plan_valid": plan_valid,
        "repair_success": repair_success,
        "command_valid": command_valid,
        "agent_trace_valid": agent_trace_valid,
        "artifact_complete": artifact_complete,
        "dry_run_coverage": dry_run_coverage,
        "trace_steps": len(response.agent_trace),
        "validation": response.validation.status,
    }


def _assignment_accuracy(response: PipelineResponse, expect: dict[str, Any]) -> float:
    expected = expect.get("expected_assignments", {})
    if not expected:
        return 1.0

    objects_by_color = {
        obj.attributes.color: obj.object_id
        for obj in response.object_table.objects
        if obj.attributes.color
    }
    assignments_by_object = {
        assignment.object_id: assignment.target
        for assignment in response.plan.assignments
    }
    correct = 0
    for color, target in expected.items():
        object_id = objects_by_color.get(color)
        if object_id and assignments_by_object.get(object_id) == target:
            correct += 1
    return correct / max(len(expected), 1)


def _agent_trace_valid(response: PipelineResponse) -> bool:
    required = {
        "segment_tool",
        "perceive_tool",
        "validate_objects_tool",
        "plan_tool",
        "validate_plan_tool",
        "execution_adapter_tool",
        "output_tool",
    }
    tool_names = {step.tool_name for step in response.agent_trace}
    failed = [step for step in response.agent_trace if step.status == "failed"]
    return required.issubset(tool_names) and not failed


def _artifact_complete(response: PipelineResponse) -> bool:
    required = {
        "segmentation",
        "object_table",
        "plan",
        "execution_commands",
        "dry_run",
        "agent_trace",
        "workflow_state",
    }
    return required.issubset(response.artifacts.keys())


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0}

    def rate(key: str) -> float:
        return round(sum(1 for item in results if item.get(key)) / total, 3)

    return {
        "total": total,
        "passed": sum(1 for item in results if item["ok"]),
        "pass_rate": rate("ok"),
        "schema_valid_rate": rate("schema_valid"),
        "object_reference_valid_rate": rate("object_reference_valid"),
        "plan_validity_rate": rate("plan_valid"),
        "repair_success_rate": rate("repair_success"),
        "command_validity_rate": rate("command_valid"),
        "agent_trace_valid_rate": rate("agent_trace_valid"),
        "artifact_complete_rate": rate("artifact_complete"),
        "dry_run_coverage_rate": rate("dry_run_coverage"),
        "avg_assignment_accuracy": round(
            sum(float(item.get("assignment_accuracy", 0.0)) for item in results) / total,
            3,
        ),
        "avg_latency_ms": round(statistics.mean(float(item["latency_ms"]) for item in results), 3),
    }


def write_report(report_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eval_results.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    csv_path = report_dir / "eval_results.csv"
    if results:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    rows = [
        "# Evaluation Report",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        rows.append(f"| {key} | {value} |")
    rows.extend([
        "",
        "## Cases",
        "",
        "| Case | OK | Assign Acc | Trace | Commands | Latency ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for item in results:
        rows.append(
            f"| {item['case_id']} | {item['ok']} | {item.get('assignment_accuracy', 0.0)} | "
            f"{item.get('trace_steps', 0)} | {item.get('commands', 0)} | {item['latency_ms']} |"
        )
    (report_dir / "eval_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
