"""Write ground-truth labels and their refs files (ML PRD Listing 4).

Labels are JSON Lines, one label per line, never read by production code --
only by the evaluation notebooks (M-09, M-10).
"""
from __future__ import annotations

import json
from pathlib import Path

from ml.generator.scenarios import GroundTruthLabel


def write_labels(labels: list[GroundTruthLabel], out_dir: Path) -> Path:
    """Write labels/<label_id>.refs.csv per label and a single labels.jsonl index."""
    out_dir = Path(out_dir)
    refs_dir = out_dir / "labels"
    refs_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "labels.jsonl"
    with labels_path.open("w") as f:
        for label in labels:
            refs_file = f"labels/{label.label_id}.refs.csv"
            (out_dir / refs_file).write_text("\n".join(label.affected_txn_refs) + "\n")
            f.write(json.dumps(label.to_dict(refs_file)) + "\n")

    return labels_path


def read_labels(labels_path: Path) -> list[dict]:
    labels_path = Path(labels_path)
    with labels_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
