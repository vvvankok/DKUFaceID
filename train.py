from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.faceid.dataset import collect_samples, to_paths_and_labels
from src.faceid.modeling import save_artifact, save_embeddings_sqlite, train_classifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FaceID baseline model (embeddings + classifier)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to dataset folder: dataset/<person_name>/*.jpg",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Where to store trained artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device: cpu or cuda. Default auto.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Local FaceNet weights (.pt) to avoid internet download.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from src.faceid.embedding import FaceEmbedder, filter_by_indices

    samples = collect_samples(args.dataset_dir)
    image_paths, labels = to_paths_and_labels(samples)
    print(f"Collected images: {len(image_paths)}")
    print(f"Unique identities: {len(set(labels))}")

    embedder = FaceEmbedder(device=args.device, weights_path=args.weights_path)
    emb = embedder.extract(image_paths=image_paths, batch_size=args.batch_size)

    used_labels = filter_by_indices(labels, emb.kept_indices)
    if emb.skipped_indices:
        print(f"Skipped images without detected faces: {len(emb.skipped_indices)}")

    outputs = train_classifier(
        embeddings=emb.embeddings,
        labels=used_labels,
        test_size=args.test_size,
        random_state=args.seed,
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "faceid_classifier.joblib"
    metrics_path = output_dir / "metrics.json"
    embeddings_path = output_dir / "embeddings.npy"
    sqlite_path = output_dir / "faceid_identities.db"

    save_artifact(model_path, outputs)
    save_embeddings_sqlite(sqlite_path, emb.embeddings, used_labels)
    np.save(embeddings_path, emb.embeddings)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": outputs.metrics,
                "classification_report": outputs.report,
                "num_images_total": len(image_paths),
                "num_images_used": len(used_labels),
                "num_images_skipped": len(emb.skipped_indices),
                "num_identities": len(set(used_labels)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Training complete.")
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Embeddings DB: {sqlite_path}")


if __name__ == "__main__":
    main()
