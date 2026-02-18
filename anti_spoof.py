from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class LivenessResult:
    is_live: bool
    live_score: float
    spoof_score: float


class AntiSpoofModel:
    """
    Wrapper over a pretrained PyTorch anti-spoof model.

    Expected model output:
    - [B, 2] logits/probabilities: class 0=spoof, class 1=live
    - [B, 1] single live logit/probability
    """

    def __init__(
        self,
        model_path: Path,
        threshold: float = 0.80,
        device: str = "cpu",
        input_size: int = 128,
        margin_ratio: float = 0.20,
        min_interval_sec: float = 0.10,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Anti-spoof model not found: {model_path}. "
                "Put pretrained weights on disk and pass --anti-spoof-model-path."
            )
        if not 0.0 < threshold < 1.0:
            raise ValueError("--anti-spoof-threshold must be between 0 and 1.")
        if input_size < 32:
            raise ValueError("--anti-spoof-input-size must be >= 32.")

        self.model_path = model_path
        self.threshold = float(threshold)
        self.input_size = int(input_size)
        self.margin_ratio = float(max(0.0, margin_ratio))
        self.min_interval_sec = float(max(0.0, min_interval_sec))
        self.device = "cpu" if device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = self._load_model(model_path)
        self._last_pred_at = 0.0
        self._last_box: np.ndarray | None = None
        self._last_result: LivenessResult | None = None

    def _load_model(self, path: Path) -> Any:
        try:
            model = torch.jit.load(str(path), map_location=self.device)
            model.eval()
            if self.device == "cpu":
                model = torch.jit.optimize_for_inference(model)
            return model
        except Exception:
            loaded = torch.load(str(path), map_location=self.device)
            if isinstance(loaded, torch.nn.Module):
                loaded.eval()
                return loaded.to(self.device)
            raise RuntimeError(
                "Unsupported anti-spoof model format. "
                "Use TorchScript .pt or serialized torch.nn.Module."
            )

    def _crop_face(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1 or bh <= 1:
            return None

        pad_x = int(bw * self.margin_ratio)
        pad_y = int(bh * self.margin_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    def _close_box(self, a: np.ndarray, b: np.ndarray, tol: float = 6.0) -> bool:
        return bool(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))) <= tol)

    def predict(self, frame: np.ndarray, box: np.ndarray, now: float) -> LivenessResult:
        if (
            self._last_result is not None
            and self._last_box is not None
            and (now - self._last_pred_at) < self.min_interval_sec
            and self._close_box(box, self._last_box)
        ):
            return self._last_result

        crop = self._crop_face(frame, box)
        if crop is None or crop.size == 0:
            raise RuntimeError("Failed to crop face ROI for anti-spoof.")

        x = self._preprocess(crop)
        with torch.inference_mode():
            out = self._model(x)

        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("Anti-spoof model returned non-tensor output.")
        out = out.float().reshape(1, -1)

        if out.shape[1] == 1:
            live_score = float(torch.sigmoid(out[0, 0]).item())
            spoof_score = 1.0 - live_score
        elif out.shape[1] >= 2:
            probs = torch.softmax(out[:, :2], dim=1)
            spoof_score = float(probs[0, 0].item())
            live_score = float(probs[0, 1].item())
        else:
            raise RuntimeError(f"Unsupported anti-spoof output shape: {tuple(out.shape)}")

        result = LivenessResult(
            is_live=live_score >= self.threshold,
            live_score=live_score,
            spoof_score=spoof_score,
        )
        self._last_pred_at = now
        self._last_box = box.copy()
        self._last_result = result
        return result
