# Execution-Ready Multimodal Agent Middleware for Robotic Sorting

中文名：面向机器人分拣的 Execution-Ready 多模态 Agent 中间层

This repository is structured as a production-style middleware layer between multimodal perception and robot execution. It accepts an `image` plus an `instruction`, then emits two execution-facing artifacts:

- `object_table.json`
- `plan.json`

The point is not a single prompt. The point is a typed workflow:

```text
image + instruction
  -> segmentation
  -> perception
  -> validation
  -> planning
  -> repair
  -> execution adapter
  -> object_table.json + plan.json
```

## Architecture

1. **Schema Layer**
   Strong Pydantic contracts for `ObjectTable`, `ObjectInstance`, `Plan`, `PlanStep`, `Assignment`, and `Rule`. JSON Schema export is available through both API routes and a script.

2. **Perception Layer**
   Converts segmentation candidates and multimodal semantic hints into a normalized object table. It keeps bbox/mask references separate from semantic attributes so downstream execution can trace every object.

3. **Planning Layer**
   Parses the instruction into rules, creates assignments, generates pick/place/inspect steps, validates object references, and repairs common failure cases.

4. **Orchestration Layer**
   Runs explicit agent tools with workflow state: segmentation, perception, validation, planning, repair, execution adapter, and output. Every tool call is captured in `agent_trace.json`.

5. **Service Layer**
   FastAPI service with:

   - `GET /health`
   - `POST /perceive`
   - `POST /plan`
   - `POST /pipeline`
   - `POST /execution/commands`
   - `POST /execution/dry-run`
   - `POST /visualize/overlay`
   - `GET /integrations/langchain/tools`
   - `GET /schemas/object-table`
   - `GET /schemas/plan`
   - `GET /demo`

6. **Observability & Evaluation Layer**
   Every request receives a request id, structured logs, intermediate artifact dumps, failure dumps, agent tool traces, dry-run command reports, and a sample eval set.

7. **Execution Adapter Layer**
   Converts validated `PlanStep` objects into AnyGrasp-style command contracts and runs structural dry-run checks before handoff.

## Quick Start

```powershell
python -m pip install -e .[dev]
uvicorn grasp_agent_middleware.api:app --reload --host 127.0.0.1 --port 8000
```

Open the demo page:

```text
http://127.0.0.1:8000/demo
```

Then call:

```powershell
curl -X POST http://127.0.0.1:8000/pipeline `
  -H "Content-Type: application/json" `
  -d "@samples/requests/sort_by_color.json"
```

## Backend Switching

Copy `.env.example` to `.env` and switch:

```text
LLM_BACKEND=heuristic
SEGMENTATION_BACKEND=stub
```

or:

```text
LLM_BACKEND=openai_compatible
OPENAI_COMPAT_BASE_URL=https://api.openai.com/v1
OPENAI_COMPAT_API_KEY=...
OPENAI_COMPAT_MODEL=gpt-4.1-mini
```

The segmentation backend is intentionally behind a protocol. Replace the stub with SAM, YOLO-Seg, GroundingDINO+SAM, or an in-house adapter without changing the schema or service contract.

### Real YOLO / SAM / VLM

Install the optional vision dependencies when you want to use a local Ultralytics model:

```powershell
python -m pip install -e ".[vision]"
```

Then configure YOLO detection/segmentation:

```text
SEGMENTATION_BACKEND=yolo
ULTRALYTICS_MODEL_PATH=yolo11n.pt
ULTRALYTICS_CONF=0.25
ULTRALYTICS_DEVICE=
ENABLE_COLOR_INFERENCE=true
```

Or configure SAM through the same adapter:

```text
SEGMENTATION_BACKEND=sam
ULTRALYTICS_MODEL_PATH=sam_b.pt
ULTRALYTICS_MODEL_FAMILY=sam
```

The `ImageInput` schema accepts exactly one of:

- `image_path`
- `image_url`
- `image_b64`
- `camera_index`
- `stream_url`

Camera example:

```json
{
  "image": {
    "camera_index": 0,
    "frame_id": "camera"
  },
  "instruction": "Pick all bottles and place them into bin recycle."
}
```

For multimodal semantic extraction, switch the LLM backend:

```text
LLM_BACKEND=openai_compatible
OPENAI_COMPAT_BASE_URL=https://api.openai.com/v1
OPENAI_COMPAT_API_KEY=...
OPENAI_COMPAT_MODEL=gpt-4.1-mini
```

## CLI

Run the local heuristic pipeline:

```powershell
python -m grasp_agent_middleware.cli --request samples/requests/sort_by_color.json
python -m grasp_agent_middleware.cli --request samples/requests/sort_by_color.json --print trace
python -m grasp_agent_middleware.cli --request samples/requests/sort_by_color.json --print commands
```

Export JSON Schemas:

```powershell
python scripts/export_schemas.py --out schemas
```

Run the sample evaluation set:

```powershell
python scripts/run_eval.py --eval-set samples/eval_set.jsonl --report-dir reports
```

The evaluation harness reports:

- schema valid rate
- object reference valid rate
- assignment accuracy
- plan validity rate
- repair success rate
- command validity rate
- agent trace valid rate
- artifact complete rate
- dry-run coverage rate
- latency

Reports are written in three formats:

- `reports/eval_report.md`
- `reports/eval_results.json`
- `reports/eval_results.csv`

Each eval case can assert minimum object count, required actions, expected color-to-target assignments, and minimum trace length.

## Visualization Demo

The browser demo at `/demo` calls `/pipeline`, then renders `/visualize/overlay` as an SVG overlay. The overlay is generated from the schema-level `ObjectTable`, so it works with stub, HTTP segmentation, YOLO, or SAM outputs.

You can call the overlay endpoint directly:

```powershell
curl -X POST http://127.0.0.1:8000/visualize/overlay `
  -H "Content-Type: application/json" `
  -d "{ \"object_table\": { ... }, \"width\": 960, \"height\": 540 }"
```

## LangChain Integration

The native workflow remains the source of truth because it gives tight control over schema validation, repair, artifacts, and execution contracts. LangChain is added as an optional adapter instead of replacing the core system.

Install:

```powershell
python -m pip install -e ".[agent]"
```

Inspect the exposed tools:

```text
GET /integrations/langchain/tools
```

Available LangChain `StructuredTool` wrappers:

- `robotic_sorting_pipeline`
- `robotic_perceive`
- `robotic_plan`
- `robotic_execution_commands`
- `robotic_execution_dry_run`

## Execution Contract

The `Plan` output is designed to be consumed by AnyGrasp-style execution code. Each object is referenced by stable `object_id`, and each step carries structured arguments instead of prose-only reasoning.

The execution adapter additionally emits `anygrasp_commands.json`, where each command links back to a `PlanStep`, `object_id`, bbox/mask reference, frame id, and optional target.

## Why This Is An Agent Project

The core system is not a single prompt. It is a tool-using agent loop with replayable decisions:

```text
segment_tool
perceive_tool
validate_objects_tool
plan_tool
validate_plan_tool
repair_tool
execution_adapter_tool
output_tool
```

Each tool call records why it was selected or skipped, how long it took, what it consumed, and what it produced. This makes failures debuggable and lets the same pipeline support offline evaluation, local demos, and future robot execution.

## Interview Talking Points

1. The project converts multimodal perception and natural-language instructions into execution-ready schemas instead of free-form text.
2. The segmentation backend is swappable: local stub for tests, HTTP adapter for external services, and optional YOLO/SAM for real images or camera streams.
3. The workflow is a tool-using agent loop with explicit state, validation, repair, trace, and artifact persistence.
4. The execution adapter emits AnyGrasp-style references, so the output is designed for robot handoff.
5. The eval harness checks schema validity, object references, plan validity, command dry-run coverage, trace completeness, and assignment accuracy.
6. LangChain is supported as an integration layer, while the production workflow stays framework-independent and easier to debug.

## Current Limits And Next Steps

- Generic YOLO weights may not recognize domain-specific sorting objects; custom fine-tuning or open-vocabulary detection would improve real scenes.
- Color inference is a lightweight HSV crop heuristic and can fail under difficult lighting.
- The execution adapter currently emits AnyGrasp-style command contracts; the next step is connecting a real robot controller and grasp pose estimator.
- Eval is mostly contract and behavior based; a stronger benchmark would add real images, bbox IoU, mask quality, and robot success rate.
