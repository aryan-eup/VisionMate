"""Expression-recognition CNN: architecture, preprocessing and checkpoint helpers."""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

IMAGE_SIZE = 48
NUM_CLASSES = 7


class ModelFileError(RuntimeError):
    """Raised when a saved model cannot be found or loaded."""


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def preprocess(images):
    """Turn uint8 grayscale faces into model input.

    Accepts one image (48, 48) or a batch (N, 48, 48) and returns a float
    tensor of shape (N, 1, 48, 48) with values scaled to [-1, 1].
    """
    array = np.asarray(images)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected faces of shape (N, {IMAGE_SIZE}, {IMAGE_SIZE}), got {array.shape}"
        )
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0).unsqueeze(1)
    return (tensor - 0.5) / 0.5


def _conv_block(in_channels, out_channels, dropout):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class ExpressionCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, channels=(64, 128, 256), dropout=0.25):
        super().__init__()
        blocks, in_channels = [], 1
        for out_channels in channels:
            blocks.append(_conv_block(in_channels, out_channels, dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)        # 48 -> 24 -> 12 -> 6
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def save_checkpoint(model, class_names, path, metrics=None):
    """Save weights plus everything needed to use them later.

    `metrics` must contain only plain Python values (floats, ints, strings, lists).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "class_names": list(class_names),
            "image_size": IMAGE_SIZE,
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(path, device=None):
    """Load a saved model. Returns (model in eval mode, class_names, metrics)."""
    path = Path(path)
    if not path.exists():
        raise ModelFileError(
            f"Model file not found: {path}. Train the model first with training/train.py."
        )

    device = device or pick_device()
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        class_names = list(data["class_names"])
        model = ExpressionCNN(num_classes=len(class_names))
        model.load_state_dict(data["state_dict"])
    except Exception as error:
        raise ModelFileError(f"Could not load {path}: {error}") from error

    model.to(device).eval()
    return model, class_names, data.get("metrics", {})


# ---------------------------------------------------------------- checks

def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def _run_checks():
    class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    device = pick_device()
    print(f"Device: {device}")

    model = ExpressionCNN().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameter_count:,}")

    # 1. Preprocessing shape and range.
    raw = np.random.default_rng(0).integers(0, 256, size=(4, 48, 48), dtype=np.uint8)
    batch = preprocess(raw)
    assert batch.shape == (4, 1, 48, 48), batch.shape
    assert batch.min() >= -1.0 and batch.max() <= 1.0
    assert preprocess(raw[0]).shape == (1, 1, 48, 48)
    try:
        preprocess(np.zeros((10, 10), dtype=np.uint8))
        raise AssertionError("preprocess accepted a wrongly sized image")
    except ValueError:
        pass
    print("Check 1 passed: preprocessing")

    # 2. Forward pass.
    model.eval()
    with torch.no_grad():
        output = model(batch.to(device))
    assert output.shape == (4, 7), output.shape
    assert torch.isfinite(output).all()
    print("Check 2 passed: forward pass, output shape (4, 7)")

    # 3. Save and load give identical predictions.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "test_model.pt"
        save_checkpoint(model, class_names, path, {"val_accuracy": 0.5})
        loaded, names, metrics = load_checkpoint(path, device)
        with torch.no_grad():
            reloaded_output = loaded(batch.to(device))
        assert names == class_names
        assert metrics == {"val_accuracy": 0.5}
        assert torch.allclose(output.cpu(), reloaded_output.cpu(), atol=1e-5)
    try:
        load_checkpoint(Path("does_not_exist.pt"))
        raise AssertionError("loading a missing file did not fail")
    except ModelFileError:
        pass
    print("Check 3 passed: save and load round trip")

    # 4. Speed of live inference (one face).
    single = batch[:1].to(device)
    with torch.no_grad():
        for _ in range(5):
            model(single)
        _sync(device)
        start = time.perf_counter()
        for _ in range(50):
            model(single)
        _sync(device)
    per_face_ms = (time.perf_counter() - start) / 50 * 1000
    print(f"Inference: {per_face_ms:.1f} ms per face")

    # 5. Speed of a training step, to estimate training time.
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(128, 1, 48, 48, device=device)
    y = torch.randint(0, 7, (128,), device=device)

    def train_step():
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()

    for _ in range(3):
        train_step()
    _sync(device)
    start = time.perf_counter()
    for _ in range(10):
        train_step()
    _sync(device)
    step_seconds = (time.perf_counter() - start) / 10
    epoch_seconds = step_seconds * (25838 / 128)
    print(f"Training step (batch 128): {step_seconds * 1000:.0f} ms "
          f"-> about {epoch_seconds:.0f} s per epoch")

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)