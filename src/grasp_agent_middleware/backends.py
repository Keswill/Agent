from __future__ import annotations

import base64
import json
import mimetypes
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from grasp_agent_middleware.config import Settings
from grasp_agent_middleware.schemas import (
    BoundingBox,
    ImageInput,
    MaskRef,
    ObjectAttributes,
    SegmentationCandidate,
)


class SegmentationBackend(Protocol):
    name: str

    def segment(self, image: ImageInput, instruction: str, request_id: str) -> list[SegmentationCandidate]:
        ...


class LLMBackend(Protocol):
    name: str

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        json_schema: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        ...


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

_SHAPES = ["cube", "block", "box", "cylinder", "bottle", "cup", "can", "ball", "sphere"]


def _tokens_from_instruction(instruction: str) -> list[tuple[str, str | None]]:
    text = instruction.lower()
    positioned_tokens: list[tuple[int, str, str | None]] = []
    for shape in _SHAPES:
        for match in re.finditer(rf"\b{re.escape(shape)}s?\b", text):
            prefix = text[max(0, match.start() - 48) : match.start()]
            color = _nearest_color(prefix)
            positioned_tokens.append((match.start(), shape, color))
    tokens = [(shape, color) for _, shape, color in sorted(positioned_tokens)]
    if not tokens:
        for color in _COLORS:
            if re.search(rf"\b{color}\b", text):
                tokens.append(("object", color))
    return tokens


def _nearest_color(text: str) -> str | None:
    matches: list[tuple[int, str]] = []
    for color in _COLORS:
        for match in re.finditer(rf"\b{color}\b", text):
            matches.append((match.start(), color))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


@dataclass
class StubSegmentationBackend:
    """Deterministic adapter used before a real segmentation model is wired in."""

    name: str = "stub"

    def segment(self, image: ImageInput, instruction: str, request_id: str) -> list[SegmentationCandidate]:
        tokens = _tokens_from_instruction(instruction)
        if not tokens:
            tokens = [("object", None)]

        candidates: list[SegmentationCandidate] = []
        width = max(len(tokens), 1)
        for index, (label, color) in enumerate(tokens, start=1):
            x_min = (index - 1) / width
            x_max = min(index / width, 1.0)
            object_label = f"{color} {label}".strip() if color else label
            candidates.append(
                SegmentationCandidate(
                    segment_id=f"seg_{index:03d}",
                    label_hint=object_label,
                    bbox=BoundingBox(
                        x_min=max(x_min + 0.03, 0.0),
                        y_min=0.18,
                        x_max=max(min(x_max - 0.03, 1.0), x_min + 0.08),
                        y_max=0.82,
                    ),
                    mask=MaskRef(uri=f"artifact://{request_id}/masks/seg_{index:03d}", encoding="none"),
                    confidence=0.45,
                    attributes=ObjectAttributes(color=color, shape=label if label != "object" else None),
                )
            )
        return candidates


@dataclass
class HeuristicLLMBackend:
    """Local no-network backend that keeps the pipeline executable in demos/tests."""

    name: str = "heuristic"

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        json_schema: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "notes": [
                "Heuristic backend does not call an external model.",
                "Use openai_compatible for multimodal semantic extraction.",
            ],
            "payload_keys": sorted(user_payload.keys()),
        }


@dataclass
class HttpSegmentationBackend:
    endpoint: str
    timeout_seconds: float = 60.0
    name: str = "http"

    def segment(self, image: ImageInput, instruction: str, request_id: str) -> list[SegmentationCandidate]:
        if not self.endpoint:
            raise RuntimeError("SEGMENTATION_ENDPOINT is required for http segmentation backend.")

        payload = {
            "request_id": request_id,
            "image": image.model_dump(mode="json"),
            "instruction": instruction,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_segments = data.get("segments", data) if isinstance(data, dict) else data
        if not isinstance(raw_segments, list):
            raise RuntimeError("HTTP segmentation backend must return a list or {'segments': [...]} payload.")
        return [SegmentationCandidate.model_validate(item) for item in raw_segments]


@dataclass
class UltralyticsSegmentationBackend:
    model_path: str
    model_family: str = "yolo"
    conf: float = 0.25
    device: str = ""
    infer_color: bool = False
    name: str = "ultralytics"

    def __post_init__(self) -> None:
        try:
            if self.model_family == "sam":
                from ultralytics import SAM

                self.model = SAM(self.model_path)
            else:
                from ultralytics import YOLO

                self.model = YOLO(self.model_path)
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics backend requires optional vision dependencies. "
                'Install with: python -m pip install -e ".[vision]"'
            ) from exc

    def segment(self, image: ImageInput, instruction: str, request_id: str) -> list[SegmentationCandidate]:
        source = _image_input_to_ultralytics_source(image)
        kwargs: dict[str, Any] = {
            "source": source,
            "stream": bool(image.stream_url or image.camera_index is not None),
        }
        if self.conf is not None and self.model_family != "sam":
            kwargs["conf"] = self.conf
        if self.device:
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)
        result = next(iter(results)) if kwargs["stream"] else results[0]
        return self._result_to_segments(result, request_id=request_id, image=image)

    def _result_to_segments(self, result: Any, *, request_id: str, image: ImageInput) -> list[SegmentationCandidate]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", {}) or {}
        orig_shape = getattr(result, "orig_shape", None) or (1, 1)
        height, width = int(orig_shape[0]), int(orig_shape[1])
        masks = getattr(result, "masks", None)
        mask_polygons = getattr(masks, "xyn", None) if masks is not None else None

        segments: list[SegmentationCandidate] = []
        for index, box in enumerate(boxes, start=1):
            cls_id = _tensor_scalar(getattr(box, "cls", [0]), int, default=0)
            confidence = _tensor_scalar(getattr(box, "conf", [self.conf]), float, default=self.conf)
            label = str(names.get(cls_id, "object"))
            x1, y1, x2, y2 = _xyxy(box)
            bbox = BoundingBox(
                x_min=_clamp(x1 / max(width, 1)),
                y_min=_clamp(y1 / max(height, 1)),
                x_max=_clamp(x2 / max(width, 1)),
                y_max=_clamp(y2 / max(height, 1)),
            )
            color, shape_label = _split_color_label(label)
            if self.infer_color and color is None:
                color = _infer_color_from_image(image, bbox)
            mask = None
            if mask_polygons is not None and index - 1 < len(mask_polygons):
                mask = MaskRef(
                    uri=f"artifact://{request_id}/masks/seg_{index:03d}.polygon",
                    encoding="polygon",
                    confidence=confidence,
                )

            segments.append(
                SegmentationCandidate(
                    segment_id=f"seg_{index:03d}",
                    label_hint=label,
                    bbox=bbox,
                    mask=mask,
                    confidence=confidence,
                    attributes=ObjectAttributes(
                        color=color,
                        shape=shape_label or label,
                        open_vocab={
                            "backend": self.name,
                            "model_family": self.model_family,
                            "model_path": self.model_path,
                            "class_id": cls_id,
                        },
                    ),
                )
            )
        return segments


@dataclass
class OpenAICompatibleLLMBackend:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0
    name: str = "openai_compatible"

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        json_schema: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_COMPAT_API_KEY is required for openai_compatible backend.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _openai_user_content(user_payload)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "execution_ready_agent_payload",
                    "strict": True,
                    "schema": json_schema,
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


def build_segmentation_backend(settings: Settings) -> SegmentationBackend:
    if settings.segmentation_backend == "stub":
        return StubSegmentationBackend()
    if settings.segmentation_backend == "http":
        return HttpSegmentationBackend(endpoint=settings.segmentation_endpoint)
    if settings.segmentation_backend in {"yolo", "sam", "ultralytics"}:
        model_family = settings.ultralytics_model_family
        if settings.segmentation_backend in {"yolo", "sam"}:
            model_family = settings.segmentation_backend
        return UltralyticsSegmentationBackend(
            model_path=settings.ultralytics_model_path,
            model_family=model_family,
            conf=settings.ultralytics_conf,
            device=settings.ultralytics_device,
            infer_color=settings.enable_color_inference,
            name=settings.segmentation_backend,
        )
    raise ValueError(
        f"Unsupported SEGMENTATION_BACKEND={settings.segmentation_backend!r}. "
        "Available: stub, http, yolo, sam, ultralytics."
    )


def build_llm_backend(settings: Settings) -> LLMBackend:
    if settings.llm_backend == "heuristic":
        return HeuristicLLMBackend()
    if settings.llm_backend == "openai_compatible":
        return OpenAICompatibleLLMBackend(
            base_url=settings.openai_compat_base_url,
            api_key=settings.openai_compat_api_key,
            model=settings.openai_compat_model,
        )
    raise ValueError(
        f"Unsupported LLM_BACKEND={settings.llm_backend!r}. "
        "Available: heuristic, openai_compatible."
    )


def _openai_user_content(user_payload: dict[str, Any]) -> str | list[dict[str, Any]]:
    image_payload = user_payload.get("image")
    text_payload = json.dumps(user_payload, ensure_ascii=True)
    if not isinstance(image_payload, dict):
        return text_payload

    image_block = _image_payload_to_openai_block(image_payload)
    if image_block is None:
        return text_payload
    return [
        {
            "type": "text",
            "text": text_payload,
        },
        image_block,
    ]


def _image_payload_to_openai_block(image_payload: dict[str, Any]) -> dict[str, Any] | None:
    if image_payload.get("image_url"):
        url = str(image_payload["image_url"])
    elif image_payload.get("image_b64"):
        url = f"data:image/png;base64,{image_payload['image_b64']}"
    elif image_payload.get("image_path"):
        path = Path(str(image_payload["image_path"]))
        if not path.exists() or not path.is_file():
            return None
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        url = f"data:{media_type};base64,{encoded}"
    else:
        return None

    return {
        "type": "image_url",
        "image_url": {
            "url": url,
        },
    }


def _image_input_to_ultralytics_source(image: ImageInput) -> str | int:
    if image.camera_index is not None:
        return image.camera_index
    if image.stream_url:
        return image.stream_url
    if image.image_url:
        return image.image_url
    if image.image_path:
        return image.image_path
    if image.image_b64:
        suffix = ".png"
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        handle.write(base64.b64decode(image.image_b64))
        handle.close()
        return handle.name
    raise ValueError("ImageInput did not contain a usable source.")


def _tensor_scalar(value: Any, caster: Any, default: Any) -> Any:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list):
            value = value[0]
        return caster(value)
    except Exception:
        return default


def _xyxy(box: Any) -> tuple[float, float, float, float]:
    values = getattr(box, "xyxy", None)
    if values is None:
        return (0.0, 0.0, 1.0, 1.0)
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _split_color_label(label: str) -> tuple[str | None, str | None]:
    normalized = re.sub(r"[_-]+", " ", label.lower()).strip()
    parts = normalized.split()
    color = next((part for part in parts if part in _COLORS), None)
    shape = " ".join(part for part in parts if part != color) or None
    return color, shape


def _infer_color_from_image(image: ImageInput, bbox: BoundingBox) -> str | None:
    if not image.image_path:
        return None
    try:
        import cv2

        path = Path(image.image_path)
        if not path.exists():
            return None
        frame = cv2.imread(str(path))
        if frame is None:
            return None
        height, width = frame.shape[:2]
        x1 = int(bbox.x_min * width)
        y1 = int(bbox.y_min * height)
        x2 = int(bbox.x_max * width)
        y2 = int(bbox.y_max * height)
        crop = frame[max(y1, 0) : max(y2, y1 + 1), max(x1, 0) : max(x2, x1 + 1)]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue = float(hsv[:, :, 0].mean())
        sat = float(hsv[:, :, 1].mean())
        val = float(hsv[:, :, 2].mean())
        return _hsv_to_color_name(hue, sat, val)
    except Exception:
        return None


def _hsv_to_color_name(hue: float, sat: float, val: float) -> str | None:
    if val < 55:
        return "black"
    if sat < 35:
        return "white" if val > 190 else "gray"
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 25:
        return "orange"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue"
    if hue < 160:
        return "purple"
    return "red"
