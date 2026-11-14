"""Evaluate the trained expression CNN on the FER-2013 test set."""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.expression_model import load_checkpoint, pick_device, preprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "fer2013_processed.npz"
MODEL_FILE = PROJECT_ROOT / "models" / "expression_cnn.pt"
ASSETS_DIR = PROJECT_ROOT / "assets"
MATRIX_FILE = ASSETS_DIR / "confusion_matrix.png"
ERRORS_FILE = ASSETS_DIR / "misclassified_examples.png"
REPORT_FILE = ASSETS_DIR / "evaluation_report.txt"


@torch.no_grad()
def predict(model, images, device, batch_size=512):
    """Return (predicted class indices, class probabilities) for uint8 faces."""
    chunks = []
    for start in range(0, len(images), batch_size):
        batch = preprocess(images[start:start + batch_size]).to(device)
        chunks.append(torch.softmax(model(batch), dim=1).cpu())
    probabilities = torch.cat(chunks).numpy()
    return probabilities.argmax(1), probabilities


def top_confusions(matrix, class_names, count=6):
    """Most common mistakes as (true, predicted, number, share of true class)."""
    mistakes = []
    for true_index in range(len(class_names)):
        row_total = matrix[true_index].sum()
        for predicted_index in range(len(class_names)):
            if true_index != predicted_index and matrix[true_index, predicted_index] > 0:
                number = int(matrix[true_index, predicted_index])
                mistakes.append((class_names[true_index], class_names[predicted_index],
                                 number, number / row_total))
    mistakes.sort(key=lambda item: item[2], reverse=True)
    return mistakes[:count]


def save_confusion_matrix(matrix, class_names, accuracy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalised = matrix / matrix.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    panels = (
        (axes[0], matrix, "Confusion matrix (image counts)", "d"),
        (axes[1], normalised, "Confusion matrix (share of each true class)", ".2f"),
    )
    for axis, values, title, number_format in panels:
        axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks(range(len(class_names)))
        axis.set_yticks(range(len(class_names)))
        axis.set_xticklabels(class_names, rotation=45, ha="right")
        axis.set_yticklabels(class_names)
        for row in range(len(class_names)):
            for col in range(len(class_names)):
                color = "white" if normalised[row, col] > 0.5 else "black"
                axis.text(col, row, format(values[row, col], number_format),
                          ha="center", va="center", color=color, fontsize=9)

    fig.suptitle(f"FER-2013 test set | accuracy {accuracy:.1%}")
    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MATRIX_FILE, dpi=110)
    plt.close(fig)


def save_misclassified(images, targets, predictions, probabilities, class_names,
                       rows=4, cols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrong = np.flatnonzero(predictions != targets)
    confidence = probabilities[wrong, predictions[wrong]]
    chosen = wrong[np.argsort(-confidence)][: rows * cols]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.4))
    for axis in axes.ravel():
        axis.axis("off")
    for axis, index in zip(axes.ravel(), chosen):
        axis.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"true: {class_names[targets[index]]}\n"
            f"pred: {class_names[predictions[index]]} "
            f"({probabilities[index, predictions[index]]:.2f})",
            fontsize=8,
        )

    fig.suptitle("Most confident mistakes (many are arguably label noise)")
    fig.tight_layout()
    fig.savefig(ERRORS_FILE, dpi=110)
    plt.close(fig)


def main():
    for path, hint in ((DATA_FILE, "python training/prepare_data.py"),
                       (MODEL_FILE, "python -m training.train")):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: {hint}")

    data = np.load(DATA_FILE, allow_pickle=False)
    x_test, y_test = data["x_test"], data["y_test"]
    data_class_names = [str(name) for name in data["class_names"]]

    device = pick_device()
    model, class_names, metrics = load_checkpoint(MODEL_FILE, device)
    if class_names != data_class_names:
        raise RuntimeError(
            f"Class mismatch: model {class_names} vs data {data_class_names}"
        )

    print(f"Evaluating on {len(y_test)} test images (device: {device})...")
    predictions, probabilities = predict(model, x_test, device)

    accuracy = float((predictions == y_test).mean())
    macro_f1 = float(f1_score(y_test, predictions, average="macro"))
    matrix = confusion_matrix(y_test, predictions, labels=range(len(class_names)))
    report = classification_report(y_test, predictions, target_names=class_names, digits=3)
    confusions = top_confusions(matrix, class_names)

    lines = [
        f"Test accuracy: {accuracy:.3f} ({accuracy:.1%})",
        f"Macro F1:      {macro_f1:.3f}",
        "",
        report,
        "Most common mistakes:",
    ]
    lines += [f"  {true:<9} -> {predicted:<9} {number:>4} images "
              f"({share:.0%} of all '{true}' faces)"
              for true, predicted, number, share in confusions]
    text = "\n".join(lines)

    print()
    print(text)

    reported = metrics.get("test_accuracy")
    if reported is not None:
        verdict = "matches" if abs(reported - accuracy) < 0.002 else "DIFFERS FROM"
        print(f"\nCross-check: {accuracy:.3f} {verdict} the {reported:.3f} reported by training.")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(text + "\n")
    save_confusion_matrix(matrix, class_names, accuracy)
    save_misclassified(x_test, y_test, predictions, probabilities, class_names)

    for path in (REPORT_FILE, MATRIX_FILE, ERRORS_FILE):
        print(f"Saved {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)