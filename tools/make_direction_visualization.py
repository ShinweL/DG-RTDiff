from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dg_rtdiff.config import load_config
from dg_rtdiff.engine import EMA, load_weights
from dg_rtdiff.models import build_model


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IDS = ("test_000258", "test_000290")
RUN_ROOT = ROOT / "runs" / "final_test" / "dg_rtdiff" / "composite"
CONFIG = ROOT / "runs_frozen" / "dg_rtdiff_final" / "config_frozen.yaml"
CHECKPOINT = ROOT / "runs" / "final_test" / "dg_rtdiff" / "checkpoint" / "best_ema.pt"
OUTPUT_PNG = ROOT / "elsarticle" / "figures" / "direction_mechanism_visualization.png"

PANEL_SIZE = 360
TITLE_HEIGHT = 42
LEFT_MARGIN = 175
RIGHT_MARGIN = 18
TOP_MARGIN = 12
ROW_GAP = 18
COL_GAP = 10


def load_signed_preview(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return array / 127.5 - 1.0


def load_signed16(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path), dtype=np.float32)
    return (array - 32768.0) / 16384.0


def percentile_limit(arrays: list[np.ndarray], percentile: float = 99.5) -> float:
    values = np.concatenate([np.abs(array).ravel() for array in arrays])
    return max(float(np.percentile(values, percentile)), 1.0e-6)


def activation_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([array.ravel() for array in arrays])
    low, high = np.percentile(values, [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def cividis_lut() -> np.ndarray:
    anchors = np.asarray(
        [
            (0x00, 0x20, 0x4D),
            (0x00, 0x33, 0x6F),
            (0x39, 0x48, 0x6B),
            (0x5F, 0x5D, 0x6A),
            (0x7B, 0x74, 0x63),
            (0x9A, 0x8C, 0x4F),
            (0xBB, 0xA6, 0x34),
            (0xDE, 0xC0, 0x00),
            (0xFF, 0xE9, 0x45),
        ],
        dtype=np.float32,
    )
    anchor_x = np.linspace(0.0, 1.0, len(anchors))
    target_x = np.linspace(0.0, 1.0, 256)
    channels = [np.interp(target_x, anchor_x, anchors[:, index]) for index in range(3)]
    return np.stack(channels, axis=1).round().astype(np.uint8)


CIVIDIS = cividis_lut()


def diverging_lut() -> np.ndarray:
    anchors = np.asarray(
        [
            (0x21, 0x66, 0xAC),
            (0x67, 0xA9, 0xCF),
            (0xF7, 0xF7, 0xF7),
            (0xEF, 0x8A, 0x62),
            (0xB3, 0x58, 0x06),
        ],
        dtype=np.float32,
    )
    anchor_x = np.linspace(0.0, 1.0, len(anchors))
    target_x = np.linspace(0.0, 1.0, 256)
    channels = [np.interp(target_x, anchor_x, anchors[:, index]) for index in range(3)]
    return np.stack(channels, axis=1).round().astype(np.uint8)


DIVERGING = diverging_lut()


def signed_panel(array: np.ndarray, limit: float) -> Image.Image:
    scaled = np.clip((array / limit + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def feature_panel(array: np.ndarray, low: float, high: float) -> Image.Image:
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    indices = np.rint(scaled * 255.0).astype(np.uint8)
    return Image.fromarray(CIVIDIS[indices], mode="RGB")


def deviation_panel(array: np.ndarray, limit: float) -> Image.Image:
    scaled = np.clip((array / limit + 1.0) * 0.5, 0.0, 1.0)
    indices = np.rint(scaled * 255.0).astype(np.uint8)
    return Image.fromarray(DIVERGING[indices], mode="RGB")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / filename
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    left, top, right, bottom = area
    box = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=3)
    width = box[2] - box[0]
    height = box[3] - box[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2 - box[1]
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=3)


@torch.no_grad()
def extract_features(model: torch.nn.Module, image: torch.Tensor) -> dict[str, np.ndarray]:
    conditioner = model.conditioner
    if conditioner is None:
        raise RuntimeError("The selected model does not contain a directional conditioner.")

    captured: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {}

    def gate_hook(_module, _inputs, output):
        captured["gate_logits"] = output.detach()

    def block_hook(_module, _inputs, output):
        captured["block_output"] = tuple(item.detach() for item in output)

    gate_handle = conditioner.blocks[0].gate.register_forward_hook(gate_hook)
    block_handle = conditioner.blocks[0].register_forward_hook(block_hook)
    encoded = model.encode_condition(image)
    gate_handle.remove()
    block_handle.remove()

    if encoded is None:
        raise RuntimeError("Directional encoding returned no features.")
    horizontal, detail, _ = captured["block_output"]
    gates = torch.softmax(captured["gate_logits"], dim=1)
    return {
        "trend": encoded["horizontal_prior"][0, 0].cpu().numpy(),
        "detail": encoded["detail_prior"][0, 0].cpu().numpy(),
        "horizontal_activation": horizontal[0].abs().mean(dim=0).cpu().numpy(),
        "detail_activation": detail[0].abs().mean(dim=0).cpu().numpy(),
        "gate_h": gates[0, 0].cpu().numpy(),
        "gate_v": gates[0, 1].cpu().numpy(),
        "gate_l": gates[0, 2].cpu().numpy(),
    }


def main() -> None:
    config = load_config(CONFIG)
    model = build_model(config["model"]).cpu()
    ema = EMA(model, float(config["training"].get("ema_decay", 0.9999)))
    load_weights(CHECKPOINT, model, ema)
    ema.apply(model)
    model.eval()

    records: list[dict[str, np.ndarray | str]] = []
    for sample_id in SAMPLE_IDS:
        sample_dir = RUN_ROOT / sample_id
        input_array = load_signed_preview(sample_dir / "input.png")
        input_tensor = torch.from_numpy(input_array).unsqueeze(0).unsqueeze(0)
        records.append(
            {
                "sample_id": sample_id,
                "input": input_array,
                "output": load_signed16(sample_dir / "target_signed16.png"),
                "reference": load_signed_preview(sample_dir / "target_truth.png"),
                **extract_features(model, input_tensor),
            }
        )

    signed_limits = {
        key: percentile_limit([record[key] for record in records])
        for key in ("input", "trend", "detail")
    }
    output_reference_limit = percentile_limit(
        [record[key] for record in records for key in ("output", "reference")]
    )
    activation_ranges = {
        key: activation_limits([record[key] for record in records])
        for key in ("horizontal_activation", "detail_activation")
    }
    gate_limits = {
        key: percentile_limit([record[key] - (1.0 / 3.0) for record in records])
        for key in ("gate_h", "gate_v", "gate_l")
    }

    cell_height = TITLE_HEIGHT + PANEL_SIZE
    width = LEFT_MARGIN + 5 * PANEL_SIZE + 4 * COL_GAP + RIGHT_MARGIN
    height = TOP_MARGIN + 4 * cell_height + 3 * ROW_GAP + 12
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(27, bold=True)
    row_font = load_font(25, bold=True)
    id_font = load_font(21)

    signed_titles = ("Input x", "Trend h", "Detail d", "DG-RTDiff output", "Reference")
    feature_titles = ("|H'1|", "|D'1|", "gH - 1/3", "gV - 1/3", "gL - 1/3")
    signed_keys = ("input", "trend", "detail", "output", "reference")
    feature_keys = (
        "horizontal_activation",
        "detail_activation",
        "gate_h",
        "gate_v",
        "gate_l",
    )

    for sample_index, record in enumerate(records):
        signed_row = sample_index * 2
        feature_row = signed_row + 1
        for row, titles, keys in (
            (signed_row, signed_titles, signed_keys),
            (feature_row, feature_titles, feature_keys),
        ):
            top = TOP_MARGIN + row * (cell_height + ROW_GAP)
            for column, (title, key) in enumerate(zip(titles, keys)):
                left = LEFT_MARGIN + column * (PANEL_SIZE + COL_GAP)
                centered_text(
                    draw,
                    (left, top, left + PANEL_SIZE, top + TITLE_HEIGHT),
                    title,
                    title_font,
                )
                if key in signed_keys:
                    limit = (
                        output_reference_limit
                        if key in {"output", "reference"}
                        else signed_limits[key]
                    )
                    panel = signed_panel(record[key], limit)
                elif key in activation_ranges:
                    low, high = activation_ranges[key]
                    panel = feature_panel(record[key], low, high)
                else:
                    panel = deviation_panel(record[key] - (1.0 / 3.0), gate_limits[key])
                panel = panel.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.BILINEAR)
                canvas.paste(panel, (left, top + TITLE_HEIGHT))

        signed_top = TOP_MARGIN + signed_row * (cell_height + ROW_GAP)
        feature_top = TOP_MARGIN + feature_row * (cell_height + ROW_GAP)
        centered_text(
            draw,
            (0, signed_top + TITLE_HEIGHT, LEFT_MARGIN - 8, signed_top + cell_height),
            f"Sample {chr(65 + sample_index)}",
            row_font,
        )
        centered_text(
            draw,
            (0, signed_top + TITLE_HEIGHT + 52, LEFT_MARGIN - 8, signed_top + cell_height + 52),
            str(record["sample_id"]),
            id_font,
        )
        centered_text(
            draw,
            (0, feature_top + TITLE_HEIGHT, LEFT_MARGIN - 8, feature_top + cell_height),
            "Learned\nfeatures",
            row_font,
        )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PNG, dpi=(300, 300), optimize=True)
    print(f"Selected samples: {', '.join(SAMPLE_IDS)}")
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()