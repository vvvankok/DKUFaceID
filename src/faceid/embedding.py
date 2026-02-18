from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

from src.faceid.weights import resolve_weights_path


@dataclass
class EmbeddingResult:
    embeddings: np.ndarray
    kept_indices: List[int]
    skipped_indices: List[int]


class FaceEmbedder:
    def __init__(self, device: str | None = None, weights_path: Path | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
        self.resnet = self._build_resnet(weights_path=weights_path).eval().to(self.device)

    def _build_resnet(self, weights_path: Path | None) -> InceptionResnetV1:
        try:
            resolved = resolve_weights_path(weights_path)
        except Exception as exc:
            raise RuntimeError(
                "Cannot prepare FaceNet weights. "
                "Set FACEID_WEIGHTS_PATH or pass --weights-path with local .pt file."
            ) from exc

        model = InceptionResnetV1(pretrained=None, classify=False)
        state = torch.load(resolved, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model

    def _embed_batch(self, faces: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            vectors = self.resnet(faces.to(self.device))
        vectors = vectors.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-8, None)
        return vectors

    def extract(self, image_paths: Sequence[Path], batch_size: int = 32) -> EmbeddingResult:
        all_embeddings: List[np.ndarray] = []
        kept_indices: List[int] = []
        skipped_indices: List[int] = []

        pending_faces: List[torch.Tensor] = []
        pending_indices: List[int] = []

        for idx, path in enumerate(image_paths):
            face = self._extract_face(path)
            if face is None:
                skipped_indices.append(idx)
                continue
            pending_faces.append(face)
            pending_indices.append(idx)

            if len(pending_faces) >= batch_size:
                batch = torch.stack(pending_faces, dim=0)
                all_embeddings.append(self._embed_batch(batch))
                kept_indices.extend(pending_indices)
                pending_faces.clear()
                pending_indices.clear()

        if pending_faces:
            batch = torch.stack(pending_faces, dim=0)
            all_embeddings.append(self._embed_batch(batch))
            kept_indices.extend(pending_indices)

        if not all_embeddings:
            raise ValueError("No faces were detected in dataset images.")

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        return EmbeddingResult(
            embeddings=embeddings,
            kept_indices=kept_indices,
            skipped_indices=skipped_indices,
        )

    def _extract_face(self, image_path: Path) -> torch.Tensor | None:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            face = self.mtcnn(img)
        return face

    def extract_single(self, image: Image.Image) -> np.ndarray | None:
        face = self.mtcnn(image.convert("RGB"))
        if face is None:
            return None
        vector = self._embed_batch(face.unsqueeze(0))[0]
        return vector.astype(np.float32)

    def extract_single_from_bgr(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb)
        return self.extract_single(image)

    def detect_boxes(self, image: Image.Image) -> np.ndarray | None:
        boxes, _ = self.mtcnn.detect(image.convert("RGB"))
        return boxes


def filter_by_indices(items: Sequence[str], indices: Iterable[int]) -> List[str]:
    idx_list = list(indices)
    return [items[i] for i in idx_list]
