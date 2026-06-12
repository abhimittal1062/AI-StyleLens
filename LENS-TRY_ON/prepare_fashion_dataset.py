from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "embedding_images" / "embedding_images"
DATASET_DIR = BASE_DIR / "datasets" / "generated-fashion-yolo"
CLASSES = ["sunglasses", "hat", "jacket", "shirt", "pants", "shorts", "skirt", "dress", "bag", "shoes"]
ALIASES = {"sunglass": "sunglasses", "shoe": "shoes"}
IMAGE_SIZE = 640
VAL_RATIO = 0.2
SEED = 42


def class_from_name(path: Path) -> str | None:
    raw = path.stem.split("_")[0].lower()
    label = ALIASES.get(raw, raw)
    return label if label in CLASSES else None


def object_box(image: Image.Image) -> tuple[float, float, float, float]:
    rgba = image.convert("RGBA")
    alpha_box = rgba.getbbox()
    if alpha_box is None:
        alpha_box = (0, 0, rgba.width, rgba.height)

    left, top, right, bottom = alpha_box
    width = max(1, right - left)
    height = max(1, bottom - top)

    x_center = (left + width / 2) / rgba.width
    y_center = (top + height / 2) / rgba.height
    norm_width = width / rgba.width
    norm_height = height / rgba.height
    return x_center, y_center, norm_width, norm_height


def square_canvas(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    rgba.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    canvas = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255, 0))
    offset = ((IMAGE_SIZE - rgba.width) // 2, (IMAGE_SIZE - rgba.height) // 2)
    canvas.paste(rgba, offset, rgba)
    return canvas


def reset_dataset() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml() -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASSES))
    (DATASET_DIR / "data.yaml").write_text(
        f"path: {DATASET_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )


def build_dataset() -> dict[str, int]:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Catalog image directory not found: {SOURCE_DIR}")

    images = [path for path in SOURCE_DIR.glob("*.png") if class_from_name(path)]
    if not images:
        raise RuntimeError(f"No catalog PNG images found in {SOURCE_DIR}")

    random.seed(SEED)
    random.shuffle(images)
    reset_dataset()

    split_counts = {"train": 0, "val": 0}
    class_counts = {name: 0 for name in CLASSES}

    val_cutoff = max(1, int(len(images) * VAL_RATIO))
    val_set = set(images[:val_cutoff])

    for source in images:
        split = "val" if source in val_set else "train"
        label = class_from_name(source)
        class_id = CLASSES.index(label)

        image = Image.open(source)
        prepared = square_canvas(image)
        box = object_box(prepared)

        target_name = source.name
        image_target = DATASET_DIR / "images" / split / target_name
        label_target = DATASET_DIR / "labels" / split / f"{source.stem}.txt"

        prepared.convert("RGB").save(image_target, "JPEG", quality=95)
        label_target.write_text(
            f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n",
            encoding="utf-8",
        )

        split_counts[split] += 1
        class_counts[label] += 1

    write_yaml()
    return {"total": len(images), **split_counts, **class_counts}


if __name__ == "__main__":
    stats = build_dataset()
    print(f"Dataset written to: {DATASET_DIR}")
    print(stats)
