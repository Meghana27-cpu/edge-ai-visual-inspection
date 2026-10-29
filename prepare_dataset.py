
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

# Original NEU-DET dataset
SOURCE = Path("dataset/dataset1/archive/NEU-DET")

# Converted YOLO dataset
OUTPUT = Path("dataset/processed")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_split_folders():
    """Find image and annotation folders for each original split."""
    splits = {}

    for split in ("train", "validation", "val"):
        root = SOURCE / split
        if not root.is_dir():
            continue

        image_dirs = [
            p for p in root.rglob("*")
            if p.is_dir() and p.name.lower() in {"images", "image"}
        ]
        annotation_dirs = [
            p for p in root.rglob("*")
            if p.is_dir()
            and p.name.lower() in {"annotations", "annotation"}
        ]

        if image_dirs and annotation_dirs:
            splits[split] = (image_dirs[0], annotation_dirs[0])

    return splits


def convert_xml(xml_path):
    """Convert Pascal VOC boxes into normalized YOLO labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError("Missing image dimensions")

    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))

    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")

    lines = []

    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()

        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class: {name}")

        box = obj.find("bndbox")
        if box is None:
            raise ValueError("Missing bounding box")

        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))

        # Clip boxes to the image boundaries.
        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Invalid bounding box coordinates")

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        lines.append(
            f"{CLASS_TO_ID[name]} "
            f"{x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    if not lines:
        raise ValueError("No object annotations found")

    return lines


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {SOURCE.resolve()}"
        )

    splits = find_split_folders()

    if "train" not in splits:
        raise FileNotFoundError(
            "Could not find train/images and train/annotations. "
            "Check the original dataset folder structure."
        )

    if not any(name in splits for name in ("validation", "val")):
        raise FileNotFoundError(
            "Could not find validation/images and validation/annotations."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    errors = []

    for original_split, (image_dir, annotation_dir) in splits.items():
        target_split = (
            "val" if original_split in ("validation", "val")
            else "train"
        )

        out_images = OUTPUT / "images" / target_split
        out_labels = OUTPUT / "labels" / target_split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        annotations = {
            p.stem: p for p in annotation_dir.glob("*.xml")
        }

        for image_path in image_dir.rglob("*"):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                continue

            xml_path = annotations.get(image_path.stem)

            if xml_path is None:
                errors.append(f"Missing XML: {image_path}")
                continue

            try:
                labels = convert_xml(xml_path)

                # Verify that the image can actually be opened.
                from PIL import Image
                with Image.open(image_path) as im:
                    im.verify()

                # Unique filenames avoid overwriting across splits.
                filename = image_path.name
                shutil.copy2(image_path, out_images / filename)

                label_path = out_labels / f"{image_path.stem}.txt"
                if label_path.exists():
                    raise ValueError(
                        f"Duplicate output label: {label_path}"
                    )

                label_path.write_text(
                    "\n".join(labels) + "\n",
                    encoding="utf-8"
                )

                stats[target_split] += 1
                for line in labels:
                    class_id = int(line.split()[0])
                    stats[f"class_{class_id}"] += 1

            except Exception as exc:
                errors.append(f"{image_path}: {exc}")

    yaml_text = (
        "path: processed\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(
            f"  {i}: {name}\n"
            for i, name in enumerate(CLASS_NAMES)
        )
    )

    (OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("\nDataset preparation complete.")
    print(f"Output: {OUTPUT.resolve()}")
    print(f"Training images converted: {stats['train']}")
    print(f"Validation images converted: {stats['val']}")

    print("\nAnnotated object counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {stats[f'class_{i}']}")

    if errors:
        print(f"\nWarnings/errors: {len(errors)}")
        for error in errors[:20]:
            print(" -", error)
        if len(errors) > 20:
            print(" - Additional errors omitted.")
        print("Fix these issues before training.")

    if stats["train"] == 0 or stats["val"] == 0:
        raise RuntimeError(
            "Training or validation set is empty. Do not train yet."
        )

    if errors:
        raise RuntimeError(
            "Some files could not be converted. Review the errors above."
        )

    print("\nNext: train using dataset/processed/data.yaml")


if __name__ == "__main__":
    main()
