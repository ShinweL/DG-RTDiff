from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


def positions(length: int, window: int, stride: int) -> list[int]:
    if length < window:
        raise ValueError(f"ROI length {length} is smaller than window {window}")
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen 320x320 R1 window CSV.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--roi", required=True, type=int, nargs=4, metavar=("L", "T", "R", "B"))
    parser.add_argument("--window-size", type=int, default=320)
    parser.add_argument("--stride", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    left, top, right, bottom = args.roi
    if right <= left or bottom <= top:
        raise ValueError("ROI must satisfy right>left and bottom>top")
    rows: list[dict[str, object]] = []
    images = sorted({path.resolve() for path in [*root.glob("*.png"), *root.glob("*.PNG")]})
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"ROI {args.roi} exceeds {image_path.name} size {(width, height)}")
        xs = positions(right - left, args.window_size, args.stride)
        ys = positions(bottom - top, args.window_size, args.stride)
        for y_index, y in enumerate(ys):
            for x_index, x in enumerate(xs):
                rows.append(
                    {
                        "window_id": f"{image_path.stem}_y{y_index:02d}_x{x_index:02d}",
                        "image_path": image_path.name,
                        "left": left + x,
                        "top": top + y,
                        "right": left + x + args.window_size,
                        "bottom": top + y + args.window_size,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["window_id", "image_path", "left", "top", "right", "bottom"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} windows from {len(images)} images to {args.output}")


if __name__ == "__main__":
    main()