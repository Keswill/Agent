from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, HTMLResponse

from grasp_agent_middleware.config import Settings
from grasp_agent_middleware.langchain_adapter import LangChainToolAdapter, ToolManifest
from grasp_agent_middleware.observability import configure_logging
from grasp_agent_middleware.orchestration import AgentWorkflow, build_workflow
from grasp_agent_middleware.schemas import (
    DryRunReport,
    ExecutionCommand,
    ExecutionRequest,
    HealthResponse,
    ObjectTable,
    PerceptionRequest,
    PipelineRequest,
    PipelineResponse,
    Plan,
    PlanningRequest,
    VisualizationRequest,
    json_schema_bundle,
)
from grasp_agent_middleware.visualization import object_table_overlay_svg


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    workflow = build_workflow(settings)

    app = FastAPI(
        title="Execution-Ready Multimodal Agent Middleware",
        version="0.1.0",
        description="Robotic sorting middleware that emits ObjectTable and Plan contracts.",
    )
    app.state.settings = settings
    app.state.workflow = workflow

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_env=settings.app_env,
            llm_backend=settings.llm_backend,
            segmentation_backend=settings.segmentation_backend,
            agent_integration=settings.agent_integration,
        )

    @app.post("/perceive", response_model=ObjectTable)
    def perceive(request: PerceptionRequest) -> ObjectTable:
        pipeline_request = PipelineRequest(
            image=request.image,
            instruction=request.instruction,
            request_id=request.request_id,
        )
        return _workflow(app).perceive(pipeline_request)

    @app.post("/plan", response_model=Plan)
    def plan(request: PlanningRequest) -> Plan:
        return _workflow(app).plan(request)

    @app.post("/pipeline", response_model=PipelineResponse)
    def pipeline(request: PipelineRequest) -> PipelineResponse:
        return _workflow(app).run(request)

    @app.post("/execution/commands", response_model=list[ExecutionCommand])
    def execution_commands(request: ExecutionRequest) -> list[ExecutionCommand]:
        return _workflow(app).build_execution_commands(request.object_table, request.plan)

    @app.post("/execution/dry-run", response_model=DryRunReport)
    def execution_dry_run(request: ExecutionRequest) -> DryRunReport:
        return _workflow(app).dry_run_execution(request.object_table, request.plan)

    @app.get("/demo", response_class=HTMLResponse)
    def demo() -> HTMLResponse:
        return HTMLResponse(_demo_html())

    @app.get("/demo/media")
    def demo_media(path: str = Query(...)) -> FileResponse:
        media_path = _safe_media_path(path)
        return FileResponse(media_path)

    @app.post("/visualize/overlay")
    def visualize_overlay(request: VisualizationRequest) -> Response:
        image_href = _demo_image_href(request.image) if request.image else None
        svg = object_table_overlay_svg(
            request.object_table,
            width=request.width,
            height=request.height,
            image_href=image_href,
        )
        return Response(svg, media_type="image/svg+xml")

    @app.get("/integrations/langchain/tools", response_model=ToolManifest)
    def langchain_tools() -> ToolManifest:
        return LangChainToolAdapter(_workflow(app)).manifest()

    @app.get("/schemas")
    def schemas() -> dict:
        return json_schema_bundle()

    @app.get("/schemas/object-table")
    def object_table_schema() -> dict:
        return ObjectTable.model_json_schema()

    @app.get("/schemas/plan")
    def plan_schema() -> dict:
        return Plan.model_json_schema()

    @app.get("/schemas/execution-command")
    def execution_command_schema() -> dict:
        return ExecutionCommand.model_json_schema()

    return app


def _workflow(app: FastAPI) -> AgentWorkflow:
    return app.state.workflow


def _safe_media_path(path: str) -> Path:
    root = Path.cwd().resolve()
    media_path = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        media_path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Media path is outside the workspace.") from exc
    if media_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
        raise HTTPException(status_code=415, detail="Unsupported media type.")
    if not media_path.exists() or not media_path.is_file():
        raise HTTPException(status_code=404, detail="Media file not found.")
    return media_path


def _demo_image_href(image) -> str | None:
    if image is None:
        return None
    if image.image_url:
        return image.image_url
    if image.image_path:
        path = Path(image.image_path)
        if path.exists():
            return f"/demo/media?path={quote(image.image_path)}"
    if image.image_b64:
        return f"data:image/png;base64,{image.image_b64}"
    return None


def _demo_html() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Robotic Sorting Agent Demo</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #171717; }
    main { max-width: 1280px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 28px; margin: 0 0 8px; }
    p { margin: 0 0 16px; color: #4a5565; }
    section { margin-top: 20px; }
    label { display: block; font-weight: 700; margin: 12px 0 6px; }
    input, textarea, button { font: inherit; border-radius: 6px; }
    input, textarea { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #c9d0da; background: #fff; }
    textarea { min-height: 86px; resize: vertical; }
    button { margin-top: 12px; padding: 10px 14px; border: 0; background: #176b87; color: #fff; cursor: pointer; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .panel { background: #fff; border: 1px solid #d8dee8; border-radius: 8px; padding: 14px; min-height: 180px; }
    .wide { grid-column: 1 / -1; }
    .overlay { min-height: 360px; display: grid; place-items: center; overflow: hidden; background: #f8fafc; }
    .overlay svg { width: 100%; height: auto; display: block; }
    .metrics { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
    .metric { padding: 7px 10px; border: 1px solid #d8dee8; border-radius: 6px; background: #fff; font-weight: 700; }
    pre { white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 13px; }
    .trace { display: grid; gap: 8px; }
    .trace div { border: 1px solid #d8dee8; border-radius: 6px; padding: 8px; background: #fff; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } main { padding: 16px; } }
  </style>
</head>
<body>
  <main>
    <h1>Robotic Sorting Agent</h1>
    <p>Run a schema-constrained multimodal agent pipeline and inspect the object table, plan, tool trace, and execution commands.</p>
    <section>
      <label for="imagePath">Image Path, URL, camera:0, or stream URL</label>
      <input id="imagePath" value="samples/images/demo_sort_scene.png" />
      <label for="instruction">Instruction</label>
      <textarea id="instruction">Sort the red cube to bin A and the blue cylinder to bin B.</textarea>
      <button onclick="runPipeline()">Run Pipeline</button>
    </section>
    <section class="grid">
      <div class="panel wide">
        <h2>Visual Overlay</h2>
        <div id="overlay" class="overlay">Waiting for a run.</div>
        <div id="metrics" class="metrics"></div>
      </div>
      <div class="panel"><h2>Object Table</h2><pre id="objects">Waiting for a run.</pre></div>
      <div class="panel"><h2>Plan</h2><pre id="plan">Waiting for a run.</pre></div>
      <div class="panel"><h2>Agent Trace</h2><div id="trace" class="trace">Waiting for a run.</div></div>
      <div class="panel"><h2>Execution Commands</h2><pre id="commands">Waiting for a run.</pre></div>
    </section>
  </main>
  <script>
    async function runPipeline() {
      const imageRef = document.getElementById("imagePath").value.trim();
      const instruction = document.getElementById("instruction").value.trim();
      const image = parseImageRef(imageRef);
      const response = await fetch("/pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image, instruction })
      });
      const data = await response.json();
      if (!response.ok) {
        document.getElementById("overlay").textContent = JSON.stringify(data, null, 2);
        return;
      }
      const overlayResponse = await fetch("/visualize/overlay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ object_table: data.object_table, image, width: 960, height: 540 })
      });
      document.getElementById("overlay").innerHTML = await overlayResponse.text();
      document.getElementById("metrics").innerHTML = [
        `objects: ${data.object_table.objects.length}`,
        `steps: ${data.plan.steps.length}`,
        `commands: ${data.execution_commands.length}`,
        `trace: ${data.agent_trace.length}`,
        `dry-run: ${data.dry_run ? data.dry_run.status : "none"}`
      ].map(item => `<span class="metric">${item}</span>`).join("");
      document.getElementById("objects").textContent = JSON.stringify(data.object_table, null, 2);
      document.getElementById("plan").textContent = JSON.stringify(data.plan, null, 2);
      document.getElementById("commands").textContent = JSON.stringify(data.execution_commands, null, 2);
      document.getElementById("trace").innerHTML = (data.agent_trace || []).map(item =>
        `<div><strong>${item.tool_name}</strong><br>${item.status} in ${item.latency_ms || 0} ms<br>${item.decision}</div>`
      ).join("");
    }
    function parseImageRef(value) {
      if (value.startsWith("camera:")) {
        return { camera_index: Number(value.replace("camera:", "")), frame_id: "camera" };
      }
      if (value.startsWith("rtsp://") || value.startsWith("rtmp://")) {
        return { stream_url: value, frame_id: "camera" };
      }
      if (value.startsWith("http")) {
        return { image_url: value, frame_id: "camera" };
      }
      return { image_path: value, frame_id: "camera" };
    }
  </script>
</body>
</html>
"""


app = create_app()
