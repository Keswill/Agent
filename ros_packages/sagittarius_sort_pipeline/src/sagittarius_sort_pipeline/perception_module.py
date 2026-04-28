from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-plus"


class PerceptionModule:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        instruction: Optional[str] = None,
        timeout_sec: float = 90.0,
    ):
        self.api_key = api_key or _env_first(
            "SORT_API_KEY",
            "QWEN_API_KEY",
            "DASHSCOPE_API_KEY",
            "OPENAI_COMPAT_API_KEY",
        )
        self.base_url = (
            base_url
            or os.getenv("SORT_BASE_URL")
            or os.getenv("OPENAI_COMPAT_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.getenv("SORT_MODEL") or os.getenv("OPENAI_COMPAT_MODEL") or DEFAULT_MODEL
        self.instruction = instruction or os.getenv(
            "PERCEPTION_INSTRUCTION",
            "Identify all visible sortable objects and describe their attributes.",
        )
        self.timeout_sec = timeout_sec

    def perceive(self, rgb_image) -> Dict[str, Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("An API key is required for the perception service.")

        width, height = _image_width_height(rgb_image)
        image_url = _rgb_image_to_data_url(rgb_image)
        raw = self._post_request(image_url=image_url, width=width, height=height)
        return _objects_from_response(raw, width=width, height=height)

    def _post_request(self, *, image_url: str, width: int, height: int) -> Dict[str, Any]:
        user_payload = {
            "instruction": self.instruction,
            "image_size": {"width": width, "height": height},
            "task": (
                "Detect every visible object relevant for robot sorting. "
                "For each object, return name, category, shape, color, need_fridge, "
                "confidence, notes, and a normalized bounding box."
            ),
            "output_contract": {
                "object_id_style": "obj_0, obj_1, ... will be assigned by caller",
                "position": "caller will convert bbox center to pixel (u, v)",
            },
        }
        content = [
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=True)},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return a single JSON object that matches the provided schema. "
                        "Bounding box coordinates must be normalized floats in [0, 1]."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_perception_objects",
                    "strict": True,
                    "schema": _response_schema(),
                },
            },
        }
        try:
            return self._post_chat_completion(payload)
        except RuntimeError as exc:
            message = str(exc)
            if "HTTP 400" not in message and "HTTP 422" not in message:
                raise
        payload.pop("response_format", None)
        payload["messages"][0]["content"] += " Return only JSON."
        return self._post_chat_completion(payload)

    def _post_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("Perception service HTTP {}: {}".format(exc.code, detail)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Perception service connection failed: {}".format(exc)) from exc

        data = json.loads(response_body)
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        return _parse_json_content(str(content))


def _objects_from_response(raw: Dict[str, Any], *, width: int, height: int) -> Dict[str, Dict[str, Any]]:
    raw_objects = raw.get("objects", [])
    if not isinstance(raw_objects, list):
        raise RuntimeError("Perception response must contain an objects list.")

    objects: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            continue
        label = _text(item.get("label")) or _text(item.get("name")) or "object"
        name = _clean_name(item.get("name") or label)
        category = _clean_category(item.get("category"), label)
        objects[f"obj_{index}"] = {
            "name": name,
            "category": category,
            "position": _bbox_center(item.get("bbox"), width=width, height=height),
            "shape": _text(item.get("shape")),
            "color": _text(item.get("color")),
            "need_fridge": bool(item.get("need_fridge", False)),
        }
    return objects


def _response_schema() -> Dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene_summary", "objects", "notes"],
        "properties": {
            "scene_summary": {"type": "string"},
            "notes": {"type": "array", "items": {"type": "string"}},
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "label",
                        "name",
                        "category",
                        "shape",
                        "color",
                        "need_fridge",
                        "confidence",
                        "bbox",
                        "notes",
                    ],
                    "properties": {
                        "label": {"type": "string"},
                        "name": {"type": "string"},
                        "category": nullable_string,
                        "shape": nullable_string,
                        "color": nullable_string,
                        "need_fridge": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "notes": {"type": "array", "items": {"type": "string"}},
                        "bbox": {
                            "type": ["object", "null"],
                            "additionalProperties": False,
                            "required": ["x_min", "y_min", "x_max", "y_max"],
                            "properties": {
                                "x_min": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "y_min": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "x_max": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "y_max": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                        },
                    },
                },
            },
        },
    }


def _rgb_image_to_data_url(rgb_image) -> str:
    encoded = _encode_with_cv2(rgb_image)
    media_type = "image/jpeg"
    if encoded is None:
        encoded = _encode_with_pillow(rgb_image)
        media_type = "image/png"
    return "data:{};base64,{}".format(media_type, encoded)


def _encode_with_cv2(rgb_image) -> Optional[str]:
    try:
        import cv2

        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return None
        return base64.b64encode(buffer.tobytes()).decode("ascii")
    except Exception:
        return None


def _encode_with_pillow(rgb_image) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install opencv-python or Pillow to encode rgb_image.") from exc
    image = Image.fromarray(rgb_image, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_width_height(rgb_image) -> Tuple[int, int]:
    height, width = rgb_image.shape[:2]
    return int(width), int(height)


def _bbox_center(bbox: Any, *, width: int, height: int) -> Optional[Tuple[int, int]]:
    if not isinstance(bbox, dict):
        return None
    try:
        x_min = _clamp(float(bbox["x_min"]))
        y_min = _clamp(float(bbox["y_min"]))
        x_max = _clamp(float(bbox["x_max"]))
        y_max = _clamp(float(bbox["y_max"]))
    except (KeyError, TypeError, ValueError):
        return None
    if x_min >= x_max or y_min >= y_max:
        return None
    u = round(((x_min + x_max) / 2.0) * width)
    v = round(((y_min + y_max) / 2.0) * height)
    return (u, v)


def _parse_json_content(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise RuntimeError("Expected a JSON object from the perception service.")
    return parsed


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_name(value: Any) -> str:
    text = _text(value) or "object"
    for suffix in ("_model", "-model", " model", "-shaped object", " shaped object", " object"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text or "object"


def _clean_category(value: Any, label: str) -> Optional[str]:
    text = (_text(value) or "").lower()
    if text and text not in {"unknown", "none", "null"}:
        return text
    label_text = label.lower()
    if any(word in label_text for word in ("carrot", "eggplant", "vegetable", "cucumber")):
        return "vegetable"
    if any(word in label_text for word in ("peach", "strawberry", "apple", "banana", "fruit")):
        return "fruit"
    return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
