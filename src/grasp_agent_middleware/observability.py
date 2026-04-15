from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root.handlers = [handler]


@dataclass
class ArtifactStore:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def request_dir(self, request_id: str) -> Path:
        path = self.root / request_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, request_id: str, name: str, payload: BaseModel | dict[str, Any] | list[Any]) -> str:
        path = self.request_dir(request_id) / name
        data = _to_jsonable(payload)
        path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        return str(path)

    def dump_failure(self, request_id: str, exc: BaseException, state: dict[str, Any]) -> str:
        payload = {
            "request_id": request_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "state": state,
        }
        return self.write_json(request_id, "failure.json", payload)


def event(logger: logging.Logger, request_id: str, message: str, **extra: Any) -> None:
    logger.info(message, extra={"request_id": request_id, **extra})


def _to_jsonable(payload: Any) -> Any:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, list):
        return [_to_jsonable(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _to_jsonable(value) for key, value in payload.items()}
    return payload
