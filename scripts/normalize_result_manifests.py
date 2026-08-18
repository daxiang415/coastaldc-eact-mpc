"""Replace machine-specific paths in copied result manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_MARKER = "/coastaldc-continuous"


def normalize_string(value: str) -> str:
    portable = value.replace("\\", "/")
    lower = portable.lower()
    marker_index = lower.find(PROJECT_MARKER)
    if marker_index >= 0:
        relative = portable[marker_index + len(PROJECT_MARKER):].lstrip("/")
        return relative or "."
    if len(portable) >= 3 and portable[1:3] == ":/" and lower.endswith("python.exe"):
        return "python"
    return value


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str):
        return normalize_string(value)
    return value


def main() -> None:
    changed = 0
    results_root = (ROOT / "results").resolve()
    if os.name == "nt":
        results_root = Path("\\\\?\\" + str(results_root))
    for path in sorted(results_root.rglob("*.json")):
        original = json.loads(path.read_text(encoding="utf-8"))
        cleaned = normalize(original)
        if cleaned != original:
            path.write_text(
                json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            changed += 1
    print(f"Normalized {changed} result manifests")


if __name__ == "__main__":
    main()
