from __future__ import annotations

import os
import ssl
import urllib.request
from pathlib import Path

from torch.hub import get_dir as get_torch_home

VGGFACE2_FILE = "20180402-114759-vggface2.pt"
VGGFACE2_URL = (
    "https://github.com/timesler/facenet-pytorch/releases/download/"
    "v2.2.9/20180402-114759-vggface2.pt"
)


def default_weights_path() -> Path:
    return Path("artifacts/weights") / VGGFACE2_FILE


def torch_cache_weights_path() -> Path:
    return Path(get_torch_home()) / "checkpoints" / VGGFACE2_FILE


def resolve_weights_path(weights_path: Path | None = None) -> Path:
    candidates = []
    if weights_path is not None:
        candidates.append(weights_path)

    env_path = os.getenv("FACEID_WEIGHTS_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(default_weights_path())
    candidates.append(torch_cache_weights_path())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    target = default_weights_path()
    download_weights(target)
    return target


def download_weights(target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download(VGGFACE2_URL, target_path, insecure=False)
    except Exception:
        _download(VGGFACE2_URL, target_path, insecure=True)


def _download(url: str, target_path: Path, insecure: bool) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "faceid-app/1.0"})
    if insecure:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=context) as resp, target_path.open(
            "wb"
        ) as out:
            out.write(resp.read())
        return

    with urllib.request.urlopen(req) as resp, target_path.open("wb") as out:
        out.write(resp.read())

