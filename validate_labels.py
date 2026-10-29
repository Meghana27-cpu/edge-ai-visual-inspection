
from pathlib import Path

label_dir = Path("dataset/processed/labels/train")
bad_labels = []
total_boxes = 0

for file in label_dir.glob("*.txt"):
    for line_number, line in enumerate(
        file.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue

        total_boxes += 1
        try:
            values = list(map(float, line.split()))

            valid = (
                len(values) == 5
                and values[0].is_integer()
                and 0 <= values[0] < 6
                and all(0 <= v <= 1 for v in values[1:])
                and values[3] > 0
                and values[4] > 0
            )

            if not valid:
                bad_labels.append(
                    (file.name, line_number, line)
                )

        except ValueError:
            bad_labels.append(
                (file.name, line_number, line)
            )

print("Total bounding boxes:", total_boxes)
print("Invalid bounding boxes:", len(bad_labels))

for item in bad_labels[:10]:
    print(item)
