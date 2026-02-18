from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


class AntiSpoofDataset(Dataset):
    """
    Expected structure:
    dataset/anti_spoof/live/*.jpg
    dataset/anti_spoof/spoof/*.jpg
    """

    def __init__(
        self,
        items: list[tuple[Path, int]],
        transform: transforms.Compose,
    ) -> None:
        self.transform = transform
        self.items = items

        if not self.items:
            raise RuntimeError("Dataset is empty.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, label


@dataclass
class TrainStats:
    train_loss: float
    val_loss: float
    val_acc: float
    live_recall: float
    spoof_recall: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train anti-spoof model and export TorchScript .pt"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/anti_spoof"),
        help="Dataset path with live/ and spoof/ folders.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("artifacts/anti_spoof_model.pt"),
        help="Output TorchScript model path.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=128,
        help="Input image size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience (epochs without val_loss improvement).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=4,
        help="Minimum epochs before early stopping can trigger.",
    )
    parser.add_argument(
        "--class-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use balanced sampler + class-weighted loss.",
    )
    parser.add_argument(
        "--live-weight-bias",
        type=float,
        default=1.15,
        help="Extra multiplier for LIVE class in loss (reduce false reject of real users).",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet-pretrained MobileNet backbone (recommended for small datasets).",
    )
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(use_pretrained_backbone: bool) -> nn.Module:
    weights = None
    if use_pretrained_backbone:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            weights = None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def collect_items(root: Path) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    live_dir = root / "live"
    spoof_dir = root / "spoof"
    for p in sorted(live_dir.glob("*")):
        if p.is_file():
            items.append((p, 1))  # live class index must be 1
    for p in sorted(spoof_dir.glob("*")):
        if p.is_file():
            items.append((p, 0))  # spoof class index must be 0
    if not items:
        raise RuntimeError(
            f"No images found in {root}. "
            "Collect data first: live/ and spoof/ folders."
        )
    return items


def stratified_split(
    items: list[tuple[Path, int]],
    val_split: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    by_label: dict[int, list[tuple[Path, int]]] = {0: [], 1: []}
    for item in items:
        by_label[item[1]].append(item)

    rng = random.Random(seed)
    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    for label_items in by_label.values():
        if not label_items:
            continue
        shuffled = label_items[:]
        rng.shuffle(shuffled)
        val_n = max(1, int(len(shuffled) * val_split))
        if val_n >= len(shuffled):
            val_n = max(1, len(shuffled) - 1)
        val_items.extend(shuffled[:val_n])
        train_items.extend(shuffled[val_n:])
    return train_items, val_items


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * x.size(0)
        count += x.size(0)
    return running_loss / max(1, count)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    tp_live = 0
    fn_live = 0
    tp_spoof = 0
    fn_spoof = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)
            y_cpu = y.detach().cpu()
            pred_cpu = pred.detach().cpu()
            tp_live += int(((pred_cpu == 1) & (y_cpu == 1)).sum().item())
            fn_live += int(((pred_cpu != 1) & (y_cpu == 1)).sum().item())
            tp_spoof += int(((pred_cpu == 0) & (y_cpu == 0)).sum().item())
            fn_spoof += int(((pred_cpu != 0) & (y_cpu == 0)).sum().item())
    live_recall = tp_live / max(1, tp_live + fn_live)
    spoof_recall = tp_spoof / max(1, tp_spoof + fn_spoof)
    return total_loss / max(1, total), (correct / max(1, total)), live_recall, spoof_recall


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    if not (0.05 <= args.val_split < 0.5):
        raise ValueError("--val-split should be in [0.05, 0.5).")

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    items = collect_items(args.dataset_dir)
    train_items, val_items = stratified_split(items, args.val_split, args.seed)
    if len(train_items) < 2:
        raise RuntimeError("Dataset too small. Add more images for live and spoof.")
    if not val_items:
        raise RuntimeError("Validation split is empty. Add more images.")

    train_ds = AntiSpoofDataset(train_items, transform=train_tf)
    val_ds = AntiSpoofDataset(val_items, transform=val_tf)

    train_labels = [label for _, label in train_items]
    counts = Counter(train_labels)
    class_weights = torch.tensor(
        [
            1.0 / max(1, counts.get(0, 0)),
            (1.0 / max(1, counts.get(1, 0))) * max(0.5, args.live_weight_bias),
        ],
        dtype=torch.float32,
    )
    class_weights = class_weights / class_weights.sum() * 2.0

    sampler = None
    shuffle = True
    if args.class_balance:
        sample_weights = [
            float(class_weights[label].item())
            for label in train_labels
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = make_model(args.pretrained_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_state = None
    best_stats = TrainStats(
        train_loss=0.0,
        val_loss=float("inf"),
        val_acc=0.0,
        live_recall=0.0,
        spoof_recall=0.0,
    )
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, live_recall, spoof_recall = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"live_recall={live_recall:.4f} spoof_recall={spoof_recall:.4f} lr={lr_now:.6f}"
        )
        improved = val_loss < (best_stats.val_loss - 1e-4)
        if improved:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_stats = TrainStats(
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                live_recall=live_recall,
                spoof_recall=spoof_recall,
            )
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}: no val_loss improvement for {no_improve} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best checkpoint.")

    model.load_state_dict(best_state)
    model.eval().cpu()

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(str(args.output_model))
    metrics_path = args.output_model.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_dir": str(args.dataset_dir),
                "num_train": len(train_items),
                "num_val": len(val_items),
                "num_live_train": counts.get(1, 0),
                "num_spoof_train": counts.get(0, 0),
                "best_epoch": best_epoch,
                "best_train_loss": best_stats.train_loss,
                "best_val_loss": best_stats.val_loss,
                "best_val_acc": best_stats.val_acc,
                "best_live_recall": best_stats.live_recall,
                "best_spoof_recall": best_stats.spoof_recall,
                "class_balance": args.class_balance,
                "live_weight_bias": args.live_weight_bias,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Training complete.")
    print(
        "Best metrics: "
        f"val_acc={best_stats.val_acc:.4f}, "
        f"live_recall={best_stats.live_recall:.4f}, "
        f"spoof_recall={best_stats.spoof_recall:.4f}"
    )
    print(f"Exported TorchScript model: {args.output_model}")
    print(f"Metrics JSON: {metrics_path}")
    print("Class mapping: spoof=0, live=1")


if __name__ == "__main__":
    main()
