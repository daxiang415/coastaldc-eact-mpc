"""Build a SHA-256 inventory for the release repository."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "FILE_MANIFEST_SHA256.csv"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    scan_root = ROOT.resolve()
    if os.name == "nt":
        scan_root = Path("\\\\?\\" + str(scan_root))
    output_name = OUTPUT.name
    rows = []
    for path in sorted(item for item in scan_root.rglob("*") if item.is_file()):
        relative = path.relative_to(scan_root).as_posix()
        parts = set(Path(relative).parts)
        if (
            relative == output_name
            or parts.intersection({".git", ".pytest_cache", "__pycache__", ".venv"})
            or path.suffix in {".pyc", ".pyo"}
            or relative.startswith("results/figure_data/output/")
        ):
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} entries to {OUTPUT.name}")


if __name__ == "__main__":
    main()
