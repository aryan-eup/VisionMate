"""Train the facial-expression CNN on FER-2013."""

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.expression_model import (
    ExpressionCNN, load_checkpoint, pick_device, preprocess, save_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "fer2013_processed.npz"
MODEL_FILE = PROJECT_ROOT / "models" / "expression_cnn.pt"
CURVES_FILE = PROJECT_ROOT / "assets" / "training_curves.png"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train the VisionMate expression CNN.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3, help="peak learning rate")
    parser.add_argument("--smoke", action="store_true",
                        help="quick 2-epoch run on a tiny subset to check the pipeline")
    return parser.parse_args()


def load_data(smoke):
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run: python training/prepare_data.py"
        )
    data = np.load(DATA_FILE, allow_pickle=False)
    splits = {name: (data[f"x_{name}"], data[f"y_{name}"])
              for name in ("train", "val", "test")}
    class_names = [str(name) for name in data["class_names"]]

    if smoke:
        rng = np.random.default_rng(SEED)
        for name, size in (("train", 1024), ("val", 512), ("test", 512)):
            x, y = splits[name]
            chosen = rng.choice(len(y), size=size, replace=False)
            splits[name] = (x[chosen], y[chosen])
    return splits, class_names


def to_device(x, y, device):
    return preprocess(x).to(device), torch.from_numpy(y).to(device)


def class_weights(y_train, num_classes):
    counts = np.maximum(np.bincount(y_train, minlength=num_classes), 1).astype(np.float64)
    weights = (counts.sum() / (num_classes * counts)) ** 0.5
    return torch.tensor(weights, dtype=torch.float32)


def augment(x):
    """Random augmentation for a batch of (N, 1, 48, 48) images in [-1, 1]."""
    n, device = x.size(0), x.device

    flip = torch.rand(n, device=device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)

    angle = (torch.rand(n, device=device) * 2 - 1) * math.radians(10)
    scale = 1 + (torch.rand(n, device=device) * 2 - 1) * 0.10
    shift_x = (torch.rand(n, device=device) * 2 - 1) * 0.15
    shift_y = (torch.rand(n, device=device) * 2 - 1) * 0.15

    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, shift_x
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, shift_y

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    contrast = 1 + (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * 0.2
    brightness = (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * 0.2
    return (x * contrast + brightness).clamp(-1, 1)


def train_one_epoch(model, x, y, optimizer, scheduler, loss_fn, batch_size):
    model.train()
    order = torch.randperm(len(y)).to(x.device)
    total_loss, correct = 0.0, 0

    for start in range(0, len(y), batch_size):
        indices = order[start:start + batch_size]
        batch_x, batch_y = augment(x[indices]), y[indices]

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(batch_y)
        correct += (logits.argmax(1) == batch_y).sum().item()
    return total_loss / len(y), correct / len(y)


@torch.no_grad()
def evaluate(model, x, y, loss_fn, batch_size=512):
    model.eval()
    total_loss, predictions = 0.0, []
    for start in range(0, len(y), batch_size):
        batch_x, batch_y = x[start:start + batch_size], y[start:start + batch_size]
        logits = model(batch_x)
        total_loss += loss_fn(logits, batch_y).item() * len(batch_y)
        predictions.append(logits.argmax(1))
    predictions = torch.cat(predictions)
    accuracy = (predictions == y).float().mean().item()
    return total_loss / len(y), accuracy, predictions.cpu().numpy()


def save_curves(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (loss_axis, acc_axis) = plt.subplots(1, 2, figsize=(11, 4))

    loss_axis.plot(epochs, history["train_loss"], label="train (weighted, augmented)")
    loss_axis.plot(epochs, history["val_loss"], label="validation")
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("epoch")
    loss_axis.legend()

    acc_axis.plot(epochs, history["train_acc"], label="train (augmented)")
    acc_axis.plot(epochs, history["val_acc"], label="validation")
    acc_axis.set_title("Accuracy")
    acc_axis.set_xlabel("epoch")
    acc_axis.legend()

    fig.tight_layout()
    CURVES_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CURVES_FILE, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    epochs = 2 if args.smoke else args.epochs
    model_file = (Path(tempfile.gettempdir()) / "visionmate_smoke.pt"
                  if args.smoke else MODEL_FILE)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()

    splits, class_names = load_data(args.smoke)
    x_train, y_train = to_device(*splits["train"], device)
    x_val, y_val = to_device(*splits["val"], device)
    x_test, y_test = to_device(*splits["test"], device)

    print("SMOKE TEST: tiny subset, results are meaningless." if args.smoke
          else "FULL TRAINING")
    print(f"Device: {device} | train {len(y_train)} | val {len(y_val)} | "
          f"test {len(y_test)} | epochs {epochs}")

    weights = class_weights(splits["train"][1], len(class_names))
    print("Class weights: " + ", ".join(
        f"{name} {w:.2f}" for name, w in zip(class_names, weights.tolist())))

    model = ExpressionCNN(num_classes=len(class_names)).to(device)
    train_loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
    eval_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(y_train) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.2,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, saved_any = 0.0, False
    print()

    try:
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(
                model, x_train, y_train, optimizer, scheduler,
                train_loss_fn, args.batch_size)
            val_loss, val_acc, _ = evaluate(model, x_val, y_val, eval_loss_fn)

            for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
                history[key].append(value)

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc, saved_any = val_acc, True
                save_checkpoint(model, class_names, model_file,
                                {"val_accuracy": float(val_acc), "epoch": int(epoch)})

            print(f"Epoch {epoch:02d}/{epochs} | train loss {train_loss:.3f} acc {train_acc:.3f}"
                  f" | val loss {val_loss:.3f} acc {val_acc:.3f}"
                  f" | lr {scheduler.get_last_lr()[0]:.1e} | {time.time() - start:.0f}s"
                  f"{'  *best*' if improved else ''}")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Using the best model saved so far.")

    if not saved_any:
        raise RuntimeError("No model was saved (training stopped before epoch 1 finished).")

    # Final, one-time evaluation on the untouched test set.
    best_model, _, best_metrics = load_checkpoint(model_file, device)
    _, test_acc, predictions = evaluate(best_model, x_test, y_test, eval_loss_fn)
    targets = y_test.cpu().numpy()

    recalls = {}
    print("\nPer-class recall on the test set:")
    for index, name in enumerate(class_names):
        mask = targets == index
        recalls[name] = float((predictions[mask] == index).mean())
        print(f"  {name:<10}{recalls[name]:.3f}   ({mask.sum()} images)")

    print(f"\nBest epoch: {best_metrics['epoch']} | "
          f"validation accuracy: {best_metrics['val_accuracy']:.3f}")
    print(f"TEST ACCURACY: {test_acc:.3f}   (random guessing would be about 0.14)")

    final_metrics = {**best_metrics, "test_accuracy": float(test_acc),
                     "per_class_recall": recalls}
    save_checkpoint(best_model, class_names, model_file, final_metrics)

    if args.smoke:
        model_file.unlink(missing_ok=True)
        print("\nSmoke test finished. Nothing was saved to models/.")
    else:
        save_curves(history)
        print(f"\nSaved model: {model_file.relative_to(PROJECT_ROOT)}")
        print(f"Saved curves: {CURVES_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)