"""Prepare FER-2013: validate images, split into train/val/test, cache as NumPy."""

import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "fer2013"
OUTPUT_FILE = PROJECT_ROOT / "data" / "fer2013_processed.npz"
SAMPLES_IMAGE = PROJECT_ROOT / "assets" / "fer2013_samples.png"

IMAGE_SIZE = 48
VAL_FRACTION = 0.10
SEED = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class DatasetError(RuntimeError):
    """Raised when the FER-2013 folder is missing or malformed."""


def find_dataset_root():
    if not DATA_DIR.exists():
        raise DatasetError(
            f"{DATA_DIR} does not exist. Download FER-2013 and unzip it there."
        )

    candidates = [DATA_DIR] + [p for p in sorted(DATA_DIR.iterdir()) if p.is_dir()]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise DatasetError(
        f"Could not find 'train' and 'test' folders inside {DATA_DIR}. "
        "Check the folder layout described in the instructions."
    )


def get_class_names(split_dir):
    return sorted(
        p.name for p in split_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def load_split(split_dir, class_names):
    """Load all images of one split. Returns (images, labels, skipped_count)."""
    images, labels, skipped = [], [], 0

    for label, name in enumerate(class_names):
        class_dir = split_dir / name
        if not class_dir.is_dir():
            raise DatasetError(f"Missing class folder: {class_dir}")

        files = sorted(f for f in class_dir.iterdir()
                       if f.suffix.lower() in IMAGE_EXTENSIONS)
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                skipped += 1
                continue
            if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE),
                                   interpolation=cv2.INTER_AREA)
            images.append(image)
            labels.append(label)

    if not images:
        raise DatasetError(f"No images found in {split_dir}")
    return np.stack(images).astype(np.uint8), np.array(labels, dtype=np.int64), skipped


def print_summary(class_names, y_train, y_val, y_test):
    print()
    print(f"{'class':<10}{'train':>8}{'val':>8}{'test':>8}")
    print("-" * 34)
    for index, name in enumerate(class_names):
        print(f"{name:<10}{(y_train == index).sum():>8}"
              f"{(y_val == index).sum():>8}{(y_test == index).sum():>8}")
    print("-" * 34)
    print(f"{'total':<10}{len(y_train):>8}{len(y_val):>8}{len(y_test):>8}")
    print()


def save_sample_grid(images, labels, class_names, per_class=8):
    import matplotlib
    matplotlib.use("Agg")  # save to file, never open a window
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(class_names), per_class,
                             figsize=(per_class * 1.3, len(class_names) * 1.45))

    for row, name in enumerate(class_names):
        indices = np.flatnonzero(labels == row)
        chosen = rng.choice(indices, size=per_class, replace=False)
        for col, index in enumerate(chosen):
            axis = axes[row, col]
            axis.imshow(images[index], cmap="gray", vmin=0, vmax=255)
            axis.axis("off")
            if col == 0:
                axis.set_title(name, fontsize=9, loc="left")

    fig.tight_layout()
    SAMPLES_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAMPLES_IMAGE, dpi=110)
    plt.close(fig)


def main():
    root = find_dataset_root()
    print(f"Dataset found at: {root}")

    class_names = get_class_names(root / "train")
    test_class_names = get_class_names(root / "test")
    if class_names != test_class_names:
        raise DatasetError(
            f"Class folders differ: train={class_names}, test={test_class_names}"
        )
    if len(class_names) != 7:
        raise DatasetError(f"Expected 7 classes, found {len(class_names)}: {class_names}")
    print(f"Classes: {class_names}")

    print("Loading train images...")
    x_train_full, y_train_full, skipped = load_split(root / "train", class_names)
    print(f"  {len(y_train_full)} loaded, {skipped} skipped")

    print("Loading test images...")
    x_test, y_test, skipped = load_split(root / "test", class_names)
    print(f"  {len(y_test)} loaded, {skipped} skipped")

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=VAL_FRACTION, stratify=y_train_full, random_state=SEED,
    )
    print_summary(class_names, y_train, y_val, y_test)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        class_names=np.array(class_names),
    )
    size_mb = OUTPUT_FILE.stat().st_size / 1e6
    print(f"Saved {OUTPUT_FILE.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")

    save_sample_grid(x_train, y_train, class_names)
    print(f"Saved sample preview: {SAMPLES_IMAGE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except DatasetError as error:
        print(f"Dataset error: {error}")
        sys.exit(1)