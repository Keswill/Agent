from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"
    log_level: str = "INFO"
    artifact_dir: Path = Path("artifacts")
    llm_backend: str = "heuristic"
    segmentation_backend: str = "stub"
    segmentation_endpoint: str = ""
    ultralytics_model_path: str = "yolo11n.pt"
    ultralytics_model_family: str = "yolo"
    ultralytics_conf: float = 0.25
    ultralytics_device: str = ""
    enable_color_inference: bool = False
    agent_integration: str = "native"
    openai_compat_base_url: str = "https://api.openai.com/v1"
    openai_compat_api_key: str = ""
    openai_compat_model: str = "gpt-4.1-mini"
    enable_failure_dumps: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_env=os.getenv("APP_ENV", cls.app_env),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            artifact_dir=Path(os.getenv("ARTIFACT_DIR", str(cls.artifact_dir))),
            llm_backend=os.getenv("LLM_BACKEND", cls.llm_backend),
            segmentation_backend=os.getenv("SEGMENTATION_BACKEND", cls.segmentation_backend),
            segmentation_endpoint=os.getenv("SEGMENTATION_ENDPOINT", cls.segmentation_endpoint),
            ultralytics_model_path=os.getenv(
                "ULTRALYTICS_MODEL_PATH", cls.ultralytics_model_path
            ),
            ultralytics_model_family=os.getenv(
                "ULTRALYTICS_MODEL_FAMILY", cls.ultralytics_model_family
            ),
            ultralytics_conf=_float_env("ULTRALYTICS_CONF", cls.ultralytics_conf),
            ultralytics_device=os.getenv("ULTRALYTICS_DEVICE", cls.ultralytics_device),
            enable_color_inference=_bool_env(
                "ENABLE_COLOR_INFERENCE", cls.enable_color_inference
            ),
            agent_integration=os.getenv("AGENT_INTEGRATION", cls.agent_integration),
            openai_compat_base_url=os.getenv(
                "OPENAI_COMPAT_BASE_URL", cls.openai_compat_base_url
            ),
            openai_compat_api_key=os.getenv(
                "OPENAI_COMPAT_API_KEY", cls.openai_compat_api_key
            ),
            openai_compat_model=os.getenv("OPENAI_COMPAT_MODEL", cls.openai_compat_model),
            enable_failure_dumps=_bool_env(
                "ENABLE_FAILURE_DUMPS", cls.enable_failure_dumps
            ),
        )
