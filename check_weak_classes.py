
from pathlib import Path
from collections import Counter
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

ROOT = Path("dataset/processed")
IMAGE_DIR = ROOT / "images" / "train"
LABEL_DIR = ROOT / "labels" / "train"
OUTPUT = Path("weak_classes_inspection.png")

NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

TARGET_CLASSES = {0, 4}
random.seed(42)

print("Starting dataset inspection...")

selected = []
for label_path in LABEL_DIR.glob("*.txt"):
    lines = [
        line.split()
        for line in label_path.read_text().splitlines()
        if line.strip()
    ]
    class_ids = {int(line[0]) for line in lines}

    if class_ids & TARGET_CLASSES:
        selected.append((label_path, lines))

print("Images containing target classes:", len(selected))

selected = random.sample(selected, min(8, len(selected)))

if not selected:
    raise RuntimeError("No crazing or rolled-in-scale labels found.")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, (label_path, lines) in zip(axes.flat, selected):
    image_path = IMAGE_DIR / f"{label_path.stem}.jpg"

    if not image_path.exists():
        ax.set_title(f"Missing: {image_path.name}")
        ax.axis("off")
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        ax.set_title("Image could not be read")
        ax.axis("off")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    ax.imshow(image)

    for values in lines:
        cls, x, y, bw, bh = map(float, values[:5])
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h

        rect = patches.Rectangle(
            (x1, y1), bw * w, bh * h,
            linewidth=1.5, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(0, y1 - 3), NAMES[int(cls)],
            color="yellow", fontsize=7,
            bbox=dict(facecolor="black", alpha=0.6)
        )

    ax.set_title(label_path.stem)
    ax.axis("off")

for ax in axes.flat[len(selected):]:
    ax.axis("off")

plt.tight_layout()
fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Inspection image saved to:", OUTPUT.resolve())
