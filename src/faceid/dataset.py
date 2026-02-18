from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: str


def collect_samples(dataset_dir: Path) -> List[Sample]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    samples: List[Sample] = []
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label = person_dir.name.strip()
        if not label:
            continue
        for image_path in sorted(person_dir.rglob("*")):
            if image_path.suffix.lower() in VALID_EXTENSIONS and image_path.is_file():
                samples.append(Sample(image_path=image_path, label=label))
    if not samples:
        raise ValueError(
            "No images were found. Expected structure: dataset/<person_name>/*.jpg"
        )
    return samples


def to_paths_and_labels(samples: Sequence[Sample]) -> Tuple[List[Path], List[str]]:
    return [s.image_path for s in samples], [s.label for s in samples]

