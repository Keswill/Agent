from __future__ import annotations

import argparse
import json
from pathlib import Path

from grasp_agent_middleware.config import Settings
from grasp_agent_middleware.observability import configure_logging
from grasp_agent_middleware.orchestration import build_workflow
from grasp_agent_middleware.schemas import PipelineRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the robotic sorting agent pipeline.")
    parser.add_argument("--request", required=True, help="Path to a PipelineRequest JSON file.")
    parser.add_argument(
        "--print",
        choices=["response", "object_table", "plan", "trace", "commands", "dry_run"],
        default="response",
        help="Payload to print to stdout.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    configure_logging(settings.log_level)
    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    request = PipelineRequest.model_validate(payload)
    response = build_workflow(settings).run(request)

    if args.print == "object_table":
        output = response.object_table
    elif args.print == "plan":
        output = response.plan
    elif args.print == "trace":
        output = response.agent_trace
    elif args.print == "commands":
        output = response.execution_commands
    elif args.print == "dry_run":
        output = response.dry_run
    else:
        output = response
    if isinstance(output, list):
        payload = [item.model_dump(mode="json") for item in output]
    elif output is None:
        payload = None
    else:
        payload = output.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
