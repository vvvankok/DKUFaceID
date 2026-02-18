from __future__ import annotations

from pathlib import Path

from src.faceid.weights import download_weights, resolve_weights_path


def main() -> None:
    target = Path("artifacts/weights/20180402-114759-vggface2.pt")
    if target.exists():
        print(f"Weights already exist: {target}")
        return
    print("Downloading FaceNet weights...")
    download_weights(target)
    final_path = resolve_weights_path(target)
    print(f"Done: {final_path}")


if __name__ == "__main__":
    main()

