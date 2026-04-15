from __future__ import annotations

import re
from dataclasses import dataclass

from grasp_agent_middleware.backends import LLMBackend
from grasp_agent_middleware.schemas import (
    ExecutionRef,
    ImageInput,
    ObjectAttributes,
    ObjectInstance,
    ObjectSource,
    ObjectTable,
    SegmentationCandidate,
    VisionExtraction,
)


_ALIASES = {
    "block": "cube",
    "box": "cube",
    "sphere": "ball",
    "can": "cylinder",
}

_COLORS = {
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
}


def canonicalize_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label.strip().lower())
    words = label.split()
    normalized = [_ALIASES.get(word, word) for word in words if word not in _COLORS]
    return " ".join(normalized) or label


def complete_attributes(label: str, attributes: ObjectAttributes) -> ObjectAttributes:
    words = set(label.lower().split())
    color = attributes.color
    if color is None:
        color = next((word for word in words if word in _COLORS), None)
    shape = attributes.shape
    if shape is None:
        shape = canonicalize_label(label).split()[-1] if label else None
    return attributes.model_copy(update={"color": color, "shape": shape})


@dataclass
class PerceptionService:
    llm_backend: LLMBackend

    def build_object_table(
        self,
        *,
        image: ImageInput,
        instruction: str,
        request_id: str,
        segments: list[SegmentationCandidate],
    ) -> ObjectTable:
        objects: list[ObjectInstance] = []
        uncertain: list[str] = []
        vision = self._extract_vision_hints(
            image=image,
            instruction=instruction,
            request_id=request_id,
            segments=segments,
        )

        for index, segment in enumerate(segments, start=1):
            hint = vision.objects[index - 1] if vision and index <= len(vision.objects) else None
            label = hint.label if hint else segment.label_hint or "object"
            base_attributes = segment.attributes
            if hint:
                base_attributes = base_attributes.model_copy(
                    update={
                        "color": hint.color or base_attributes.color,
                        "material": hint.material or base_attributes.material,
                        "shape": hint.shape or base_attributes.shape,
                        "state": hint.state or base_attributes.state,
                    }
                )
            attributes = complete_attributes(label, base_attributes)
            object_id = f"obj_{index:03d}"
            canonical_label = canonicalize_label(label)
            confidence = min(max(max(segment.confidence, hint.confidence if hint else 0.0), 0.0), 1.0)
            source = ObjectSource.fused if segment.label_hint or hint else ObjectSource.segmentation
            notes: list[str] = []
            if confidence < 0.55:
                uncertain.append(object_id)
                notes.append("Low-confidence object candidate; execution should inspect before pick.")

            objects.append(
                ObjectInstance(
                    object_id=object_id,
                    label=label,
                    canonical_label=canonical_label,
                    aliases=sorted({label, canonical_label} - {canonical_label}),
                    attributes=attributes,
                    bbox=segment.bbox,
                    mask=segment.mask,
                    confidence=confidence,
                    graspable=True,
                    source=source,
                    execution_ref=ExecutionRef(
                        frame_id=image.frame_id,
                        object_ref=object_id,
                        bbox=segment.bbox,
                        mask=segment.mask,
                        grasp_hints=["top_grasp", "verify_clearance"],
                    ),
                    notes=notes,
                )
            )

        scene_summary = vision.scene_summary if vision and vision.scene_summary else (
            f"Detected {len(objects)} candidate object(s) for instruction-driven sorting."
        )
        return ObjectTable(
            request_id=request_id,
            image_ref=image.display_ref(),
            objects=objects,
            scene_summary=scene_summary,
            uncertain_objects=uncertain,
            metadata={
                "llm_backend": self.llm_backend.name,
                "semantic_mode": "heuristic_or_external_backend",
                "instruction": instruction,
                "vision_notes": vision.notes if vision else [],
            },
        )

    def _extract_vision_hints(
        self,
        *,
        image: ImageInput,
        instruction: str,
        request_id: str,
        segments: list[SegmentationCandidate],
    ) -> VisionExtraction | None:
        if self.llm_backend.name == "heuristic":
            return None

        payload = {
            "image": image.model_dump(mode="json"),
            "instruction": instruction,
            "segmentation_candidates": [segment.model_dump(mode="json") for segment in segments],
            "task": (
                "Return normalized object labels and attributes for robotic sorting. "
                "Keep object order aligned with segmentation_candidates when possible."
            ),
        }
        raw = self.llm_backend.complete_json(
            system_prompt=(
                "You are a robotic sorting perception module. "
                "Return only JSON that matches the provided schema."
            ),
            user_payload=payload,
            json_schema=VisionExtraction.model_json_schema(),
            request_id=request_id,
        )
        return VisionExtraction.model_validate(raw)
