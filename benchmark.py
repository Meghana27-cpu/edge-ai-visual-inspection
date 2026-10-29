
import argparse
import statistics
import time
from pathlib import Path

import torch
from ultralytics import YOLO


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
TEST_IMAGE = (
    BASE_DIR / "dataset" / "processed" / "images" / "val" / "crazing_241.jpg"
)


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO26 inference")
    parser.add_argument("--runs", type=int, default=20, help="Number of timed runs")
    parser.add_argument(
        "--device",
        choices=["cpu", "0"],
        default="cpu",
        help="Use cpu or GPU 0",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    device = args.device

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_IMAGE.exists():
        raise FileNotFoundError(f"Test image not found: {TEST_IMAGE}")

    if device == "0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check your PyTorch installation.")

    print(f"Model: {MODEL_PATH}")
    print(f"Test image: {TEST_IMAGE}")
    print(f"Device: {'NVIDIA GPU: ' + torch.cuda.get_device_name(0) if device == '0' else 'CPU'}")

    print("Loading model...")
    model = YOLO(str(MODEL_PATH))

    def synchronize():
        if device == "0":
            torch.cuda.synchronize()

    print("Warming up...")
    for _ in range(3):
        model.predict(
            source=str(TEST_IMAGE),
            imgsz=640,
            device=device,
            verbose=False,
        )
    synchronize()

    timings = []
    last_results = None

    print(f"Running {args.runs} benchmark iterations...")
    for i in range(args.runs):
        synchronize()
        start = time.perf_counter()

        last_results = model.predict(
            source=str(TEST_IMAGE),
            imgsz=640,
            device=device,
            verbose=False,
        )

        synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)
        print(f"Run {i + 1:02d}: {elapsed_ms:.2f} ms")

    avg_ms = statistics.mean(timings)
    median_ms = statistics.median(timings)
    p95_ms = sorted(timings)[min(len(timings) - 1, int(0.95 * len(timings)))]

    detections = len(last_results[0].boxes) if last_results else 0

    print("\n===== YOLO26 BENCHMARK RESULTS =====")
    print(f"Device:                 {'GPU' if device == '0' else 'CPU'}")
    print(f"Iterations:             {args.runs}")
    print(f"Average latency:        {avg_ms:.2f} ms")
    print(f"Median latency:         {median_ms:.2f} ms")
    print(f"95th-percentile latency: {p95_ms:.2f} ms")
    print(f"Approx. throughput:     {1000 / avg_ms:.2f} images/second")
    print(f"Last run detections:    {detections}")
    print("\nTimings include the prediction call, but exclude model loading and warm-up.")


if __name__ == "__main__":
    main()
