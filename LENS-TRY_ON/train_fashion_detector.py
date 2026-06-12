from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from prepare_fashion_dataset import BASE_DIR, DATASET_DIR, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the local StyleLens fashion detector.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-model", default=str(BASE_DIR / "yolov8s.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset()
    print("Prepared dataset:", stats)

    model_path = Path(args.base_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Base YOLO model not found: {model_path}")

    model = YOLO(str(model_path))
    results = model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(BASE_DIR / "runs" / "detect"),
        name="fashion-local",
        exist_ok=True,
        patience=max(2, args.epochs),
        verbose=True,
    )

    best_source = Path(results.save_dir) / "weights" / "best.pt"
    best_target = BASE_DIR / "best.pt"
    if not best_source.exists():
        raise FileNotFoundError(f"Training completed but best.pt was not found at {best_source}")
    shutil.copy2(best_source, best_target)
    print(f"Copied trained model to: {best_target}")


if __name__ == "__main__":
    main()
