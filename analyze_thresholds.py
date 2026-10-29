
from pathlib import Path
import csv

import numpy as np
from PIL import Image
from ultralytics import YOLO


# -------------------- SETTINGS --------------------

MODEL_PATH = "models/best.pt"
DATASET_PATH = Path("dataset/processed")
IMAGE_DIR = DATASET_PATH / "images" / "val"
LABEL_DIR = DATASET_PATH / "labels" / "val"

THRESHOLDS = [
    round(x, 2) for x in np.arange(0.10, 0.76, 0.05)
]

IOU_THRESHOLD = 0.50
IMAGE_SIZE = 512
DEVICE = 0


# -------------------- IOU CALCULATION --------------------

def calculate_iou(box, boxes):
    """Calculate IoU between one box and multiple boxes."""

    if len(boxes) == 0:
        return np.array([])

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
    )

    area_box = (
        max(0, box[2] - box[0])
        * max(0, box[3] - box[1])
    )

    area_boxes = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    union = area_box + area_boxes - intersection

    return intersection / np.maximum(union, 1e-9)


# -------------------- LOAD GROUND TRUTH --------------------

def load_ground_truth(label_path, image_width, image_height):
    """
    Load YOLO labels and convert normalized xywh
    coordinates into pixel xyxy coordinates.

    Returns columns:
    class_id, x1, y1, x2, y2
    """

    if not label_path.exists():
        return np.empty((0, 5), dtype=float)

    rows = []

    for line in label_path.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        cls, xc, yc, width, height = map(float, parts)

        # Normalized YOLO coordinates -> pixel coordinates
        x1 = (xc - width / 2) * image_width
        y1 = (yc - height / 2) * image_height
        x2 = (xc + width / 2) * image_width
        y2 = (yc + height / 2) * image_height

        rows.append([int(cls), x1, y1, x2, y2])

    return np.array(rows, dtype=float).reshape(-1, 5)


# -------------------- MAIN EVALUATION --------------------

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Validation images not found: {IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Validation labels not found: {LABEL_DIR}"
        )

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not images:
        raise RuntimeError("No validation images were found.")

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Validation images: {len(images)}")
    print("Running GPU inference one image at a time...")

    # Store predictions and ground truth for each image.
    predictions = {}
    ground_truth = {}

    for index, image_path in enumerate(images, start=1):

        # Load image dimensions.
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        ground_truth[image_path.stem] = load_ground_truth(
            label_path,
            image_width,
            image_height,
        )

        # Low confidence cutoff allows us to test thresholds later.
        result = model.predict(
            source=str(image_path),
            conf=0.001,
            iou=0.7,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            max_det=300,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            predictions[image_path.stem] = np.empty((0, 6))
        else:
            predictions[image_path.stem] = np.column_stack((
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.xyxy.cpu().numpy(),
            ))

        if index % 50 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)} images")

    # -------------------- THRESHOLD SWEEP --------------------

    results = []

    print("\n--- THRESHOLD RESULTS ---")

    for threshold in THRESHOLDS:
        tp = 0
        fp = 0
        fn = 0

        for image_path in images:
            image_id = image_path.stem

            gt = ground_truth[image_id]
            pred = predictions[image_id]

            # Filter predictions by confidence.
            pred = pred[pred[:, 1] >= threshold]

            # Highest-confidence predictions are matched first.
            order = (
                np.argsort(-pred[:, 1])
                if len(pred)
                else []
            )

            matched_gt = set()

            for pred_index in order:
                predicted_class = int(pred[pred_index, 0])
                predicted_box = pred[pred_index, 2:6]

                # Ground-truth boxes of the same class
                candidates = [
                    j for j in range(len(gt))
                    if j not in matched_gt
                    and int(gt[j, 0]) == predicted_class
                ]

                if not candidates:
                    fp += 1
                    continue

                candidate_boxes = gt[candidates, 1:5]

                ious = calculate_iou(
                    predicted_box,
                    candidate_boxes,
                )

                best_index = int(np.argmax(ious))

                if ious[best_index] >= IOU_THRESHOLD:
                    tp += 1
                    matched_gt.add(candidates[best_index])
                else:
                    fp += 1

            fn += len(gt) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        row = {
            "confidence": threshold,
            "precision": precision,
            "recall": recall,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }

        results.append(row)

        print(
            f"conf={threshold:.2f} | "
            f"precision={precision:.2%} | "
            f"recall={recall:.2%} | "
            f"TP={tp} FP={fp} FN={fn}"
        )

    # -------------------- SAVE RESULTS --------------------

    output_path = Path("threshold_results.csv")

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "confidence",
                "precision",
                "recall",
                "TP",
                "FP",
                "FN",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    # Choose the highest-recall threshold with precision >= 85%.
    eligible = [
        row for row in results
        if row["precision"] >= 0.85
    ]

    print("\n--- RECOMMENDED THRESHOLD ---")

    if eligible:
        best = max(eligible, key=lambda row: row["recall"])

        print(f"Confidence: {best['confidence']:.2f}")
        print(f"Precision:  {best['precision']:.2%}")
        print(f"Recall:     {best['recall']:.2%}")
        print(f"TP: {best['TP']} | FP: {best['FP']} | FN: {best['FN']}")
    else:
        print("No tested threshold achieved 85% precision.")

    print(f"\nResults saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
