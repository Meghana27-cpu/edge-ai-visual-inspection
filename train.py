
from pathlib import Path
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "processed" / "data.yaml"
START_MODEL = BASE_DIR / "models" / "best.pt"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_YAML}")

    if not START_MODEL.exists():
        raise FileNotFoundError(f"Checkpoint not found: {START_MODEL}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check your GPU environment.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Starting from:", START_MODEL)

    model = YOLO(str(START_MODEL))

    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        patience=20,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        amp=True,

        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Moderate augmentation
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.15,
        degrees=3.0,
        translate=0.05,
        scale=0.20,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.3,
        mixup=0.0,
        close_mosaic=10,

        val=True,
        plots=True,
        save=True,
        save_period=10,
        seed=42,
        deterministic=True,

        project=str(BASE_DIR / "results"),
        name="neu_yolo26n_accuracy_v2",
        exist_ok=False,
    )

    print("Training complete.")
    print("Best checkpoint:", Path(results.save_dir) / "weights" / "best.pt")

if __name__ == "__main__":
    main()
