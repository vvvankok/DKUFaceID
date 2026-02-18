from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict identity from one image using trained classifier."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Local FaceNet weights (.pt) to avoid internet download.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from src.faceid.embedding import FaceEmbedder

    payload = joblib.load(args.model_path)
    clf = payload["classifier"]
    label_encoder = payload["label_encoder"]

    embedder = FaceEmbedder(device=args.device, weights_path=args.weights_path)
    result = embedder.extract([args.image_path], batch_size=1)
    embedding = result.embeddings[0:1]

    probs = clf.predict_proba(embedding)[0]
    class_idx = int(np.argmax(probs))
    class_id = clf.classes_[class_idx]
    label = str(label_encoder.inverse_transform([class_id])[0])
    confidence = float(probs[class_idx])

    print(f"Identity: {label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
