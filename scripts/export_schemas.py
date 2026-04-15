from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_agent_middleware.schemas import json_schema_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export JSON Schemas for middleware contracts.")
    parser.add_argument("--out", default="schemas", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, schema in json_schema_bundle().items():
        path = out_dir / f"{name}.schema.json"
        path.write_text(json.dumps(schema, ensure_ascii=True, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()

