from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "0.1.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CoordinateSpace(str, Enum):
    pixel = "pixel"
    normalized = "normalized"


class ObjectSource(str, Enum):
    segmentation = "segmentation"
    multimodal_llm = "multimodal_llm"
    fused = "fused"
    heuristic = "heuristic"
    repaired = "repaired"


class PlanAction(str, Enum):
    inspect = "inspect"
    pick = "pick"
    place = "place"
    skip = "skip"
    replan = "replan"


class RuleKind(str, Enum):
    assignment = "assignment"
    safety = "safety"
    priority = "priority"
    validation = "validation"


class ValidationStatus(str, Enum):
    passed = "passed"
    repaired = "repaired"
    failed = "failed"


class AgentToolStatus(str, Enum):
    selected = "selected"
    skipped = "skipped"
    completed = "completed"
    failed = "failed"


class ImageInput(BaseModel):
    """Image payload reference. Exactly one image source should be supplied."""

    model_config = ConfigDict(extra="forbid")

    image_url: str | None = None
    image_path: str | None = None
    image_b64: str | None = None
    camera_index: int | None = Field(default=None, ge=0)
    stream_url: str | None = None
    frame_id: str = "camera"

    @model_validator(mode="after")
    def exactly_one_image_source(self) -> "ImageInput":
        supplied = [
            self.image_url,
            self.image_path,
            self.image_b64,
            self.camera_index,
            self.stream_url,
        ]
        if sum(value is not None for value in supplied) != 1:
            raise ValueError(
                "Provide exactly one of image_url, image_path, image_b64, "
                "camera_index, or stream_url."
            )
        return self

    def display_ref(self) -> str:
        if self.image_url:
            return self.image_url
        if self.image_path:
            return self.image_path
        if self.camera_index is not None:
            return f"camera:{self.camera_index}"
        if self.stream_url:
            return self.stream_url
        return "base64:image"


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    coordinate_space: CoordinateSpace = CoordinateSpace.normalized

    @model_validator(mode="after")
    def valid_box(self) -> "BoundingBox":
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Bounding box min coordinates must be less than max coordinates.")
        if self.coordinate_space == CoordinateSpace.normalized:
            values = [self.x_min, self.y_min, self.x_max, self.y_max]
            if any(value < 0 or value > 1 for value in values):
                raise ValueError("Normalized bounding box coordinates must be in [0, 1].")
        return self


class MaskRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    encoding: Literal["rle", "polygon", "png", "npy", "none"] = "none"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ObjectAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color: str | None = None
    material: str | None = None
    shape: str | None = None
    size: str | None = None
    state: str | None = None
    text: str | None = None
    open_vocab: dict[str, Any] = Field(default_factory=dict)


class ExecutionRef(BaseModel):
    """Stable reference consumed by an execution-side grasp planner."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = "camera"
    object_ref: str
    bbox: BoundingBox | None = None
    mask: MaskRef | None = None
    grasp_hints: list[str] = Field(default_factory=list)


class ObjectInstance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str = Field(pattern=r"^obj_[0-9]{3,}$")
    label: str
    canonical_label: str
    aliases: list[str] = Field(default_factory=list)
    attributes: ObjectAttributes = Field(default_factory=ObjectAttributes)
    bbox: BoundingBox | None = None
    mask: MaskRef | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    graspable: bool = True
    source: ObjectSource = ObjectSource.fused
    execution_ref: ExecutionRef | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("label", "canonical_label")
    @classmethod
    def non_empty_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Object labels cannot be empty.")
        return value


class ObjectTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    request_id: str
    image_ref: str
    created_at: datetime = Field(default_factory=utc_now)
    objects: list[ObjectInstance] = Field(default_factory=list)
    scene_summary: str = ""
    uncertain_objects: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_object_ids(self) -> "ObjectTable":
        ids = [obj.object_id for obj in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("ObjectTable contains duplicate object_id values.")
        return self


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(pattern=r"^rule_[0-9]{3,}$")
    kind: RuleKind
    description: str
    conditions: dict[str, Any] = Field(default_factory=dict)
    action: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=100, ge=0, le=1000)


class Assignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignment_id: str = Field(pattern=r"^assign_[0-9]{3,}$")
    object_id: str = Field(pattern=r"^obj_[0-9]{3,}$")
    target: str
    rule_id: str | None = Field(default=None, pattern=r"^rule_[0-9]{3,}$")
    rationale: str = ""
    constraints: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int = Field(ge=1)
    action: PlanAction
    object_id: str | None = Field(default=None, pattern=r"^obj_[0-9]{3,}$")
    target: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    expected_result: str = ""
    on_failure: str = "replan"


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ValidationStatus
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    repaired: list[str] = Field(default_factory=list)


class AgentTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: int = Field(ge=1)
    tool_name: str
    decision: str
    status: AgentToolStatus
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    request_id: str
    instruction: str
    object_table_ref: str
    created_at: datetime = Field(default_factory=utc_now)
    rules: list[Rule] = Field(default_factory=list)
    assignments: list[Assignment] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    validation: ValidationReport = Field(
        default_factory=lambda: ValidationReport(status=ValidationStatus.passed)
    )
    execution_contract: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def steps_are_unique(self) -> "Plan":
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan contains duplicate step_id values.")
        return self


class ExecutionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_id: str = Field(pattern=r"^cmd_[0-9]{3,}$")
    step_id: int = Field(ge=1)
    action: PlanAction
    object_id: str | None = Field(default=None, pattern=r"^obj_[0-9]{3,}$")
    target: str | None = None
    frame_id: str = "camera"
    object_ref: str | None = None
    bbox: BoundingBox | None = None
    mask: MaskRef | None = None
    grasp_planner: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class DryRunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_id: str = Field(pattern=r"^cmd_[0-9]{3,}$")
    status: Literal["ready", "skipped", "blocked"]
    message: str


class DryRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: ValidationStatus
    commands_checked: int = Field(ge=0)
    events: list[DryRunEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class VisionObjectHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    color: str | None = None
    material: str | None = None
    shape: str | None = None
    state: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class VisionExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_summary: str = ""
    objects: list[VisionObjectHint] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SegmentationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    label_hint: str | None = None
    bbox: BoundingBox | None = None
    mask: MaskRef | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    attributes: ObjectAttributes = Field(default_factory=ObjectAttributes)


class PipelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: ImageInput
    instruction: str = Field(min_length=1)
    request_id: str | None = None

    @field_validator("instruction")
    @classmethod
    def clean_instruction(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Instruction cannot be blank.")
        return value


class PerceptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: ImageInput
    instruction: str = Field(min_length=1)
    request_id: str | None = None


class PlanningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    object_table: ObjectTable
    request_id: str | None = None


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_table: ObjectTable
    plan: Plan
    request_id: str | None = None


class VisualizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_table: ObjectTable
    image: ImageInput | None = None
    width: int = Field(default=960, ge=320, le=1920)
    height: int = Field(default=540, ge=240, le=1080)


class PipelineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    object_table: ObjectTable
    plan: Plan
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)
    execution_commands: list[ExecutionCommand] = Field(default_factory=list)
    dry_run: DryRunReport | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    validation: ValidationReport


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    app_env: str
    llm_backend: str
    segmentation_backend: str
    agent_integration: str


def json_schema_bundle() -> dict[str, Any]:
    return {
        "ObjectTable": ObjectTable.model_json_schema(),
        "Plan": Plan.model_json_schema(),
        "AgentTraceStep": AgentTraceStep.model_json_schema(),
        "ExecutionCommand": ExecutionCommand.model_json_schema(),
        "DryRunReport": DryRunReport.model_json_schema(),
        "ExecutionRequest": ExecutionRequest.model_json_schema(),
        "VisualizationRequest": VisualizationRequest.model_json_schema(),
        "HealthResponse": HealthResponse.model_json_schema(),
        "PipelineRequest": PipelineRequest.model_json_schema(),
        "PipelineResponse": PipelineResponse.model_json_schema(),
    }
