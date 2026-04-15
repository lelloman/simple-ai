#!/usr/bin/env python3
"""Evaluate OMR score-region detection with the published OLA YOLO model.

This script is intentionally standalone. It does not call SimpleAI APIs and it
does not require a runner. It is for collecting data points before deciding
whether to productize OMR layout detection.

Install dependencies in a throwaway environment:

    python3 -m venv .venv-omr
    . .venv-omr/bin/activate
    pip install ultralytics pillow pypdfium2

Run on images or PDFs:

    python3 scripts/evaluate_omr_layout.py scores/*.pdf --out /tmp/omr-eval
    python3 scripts/evaluate_omr_layout.py page.png --model /path/to/OLA_v2.pt

Outputs:
    detections.json   Per-page detections and timing.
    summary.csv       One row per page with counts by class.
    summary.json      Aggregate totals and latency statistics.
    annotated/        Optional annotated page images, enabled by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

RELEASE_API_URL = (
    "https://api.github.com/repos/v-dvorak/omr-layout-analysis/releases/latest"
)
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "simple-ai" / "omr" / "OLA_v2.pt"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PDF_EXTENSIONS = {".pdf"}


@dataclass(frozen=True)
class PageInput:
    source: Path
    page_index: int
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone OMR layout detection on score PDFs/images.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input image/PDF files or directories containing them.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("omr-layout-eval"),
        help="Output directory. Default: ./omr-layout-eval",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"YOLO .pt model path. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--download-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the latest OLA .pt release asset if --model is missing.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="YOLO NMS IoU threshold. Default: 0.7",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size. Default: 1280",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="YOLO device, e.g. cpu, 0, cuda:0. Default: cpu",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=200,
        help="PDF rasterization DPI. Default: 200",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to process per PDF.",
    )
    parser.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write annotated page images. Default: true",
    )
    return parser.parse_args()


def require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install with: "
            "pip install ultralytics pillow pypdfium2"
        ) from exc
    return YOLO


def require_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pillow. Install with: "
            "pip install -r scripts/omr-layout-requirements.txt"
        ) from exc
    return Image, ImageDraw, ImageFont


def require_pdfium() -> Any:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise SystemExit(
            "PDF input requires pypdfium2. Install with: "
            "pip install -r scripts/omr-layout-requirements.txt"
        ) from exc
    return pdfium


def download_model(model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading latest OLA model to {model_path}", file=sys.stderr)

    with urllib.request.urlopen(RELEASE_API_URL, timeout=30) as response:
        release = json.load(response)

    assets = release.get("assets") or []
    candidates = [
        asset
        for asset in assets
        if str(asset.get("name", "")).lower().endswith(".pt")
    ]
    if not candidates:
        raise SystemExit(
            "The latest omr-layout-analysis release did not expose a .pt asset. "
            "Download it manually from "
            "https://github.com/v-dvorak/omr-layout-analysis/releases "
            "and pass --model /path/to/model.pt."
        )

    asset = candidates[0]
    url = asset.get("browser_download_url")
    if not url:
        raise SystemExit(f"Release asset {asset.get('name')} has no download URL")

    tmp_path = model_path.with_suffix(model_path.suffix + ".download")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(model_path)


def iter_input_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            for child in sorted(item.rglob("*")):
                if child.suffix.lower() in IMAGE_EXTENSIONS | PDF_EXTENSIONS:
                    files.append(child)
        elif item.is_file():
            if item.suffix.lower() in IMAGE_EXTENSIONS | PDF_EXTENSIONS:
                files.append(item)
            else:
                print(f"Skipping unsupported file: {item}", file=sys.stderr)
        else:
            print(f"Skipping missing path: {item}", file=sys.stderr)
    return files


def rasterize_pdf(pdf_path: Path, out_dir: Path, dpi: int, max_pages: int | None) -> list[PageInput]:
    pdfium = require_pdfium()
    pages: list[PageInput] = []
    scale = dpi / 72.0
    document = pdfium.PdfDocument(str(pdf_path))
    page_count = len(document)
    if max_pages is not None:
        page_count = min(page_count, max_pages)

    pdf_dir = out_dir / safe_stem(pdf_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for page_index in range(page_count):
        page = document[page_index]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        image_path = pdf_dir / f"page-{page_index + 1:04d}.png"
        image.save(image_path)
        pages.append(PageInput(pdf_path, page_index, image_path))
    return pages


def collect_pages(files: list[Path], work_dir: Path, pdf_dpi: int, max_pages: int | None) -> list[PageInput]:
    pages: list[PageInput] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            pages.append(PageInput(file_path, 0, file_path))
        elif suffix in PDF_EXTENSIONS:
            pages.extend(rasterize_pdf(file_path, work_dir, pdf_dpi, max_pages))
    return pages


def safe_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in path.stem)


def detection_color(name: str) -> tuple[int, int, int]:
    palette = {
        "staff": (35, 139, 69),
        "staves": (35, 139, 69),
        "stave": (35, 139, 69),
        "staffmeasure": (33, 113, 181),
        "staff_measure": (33, 113, 181),
        "stave_measure": (33, 113, 181),
        "grandstaff": (128, 0, 128),
        "grand_staff": (128, 0, 128),
        "system": (217, 95, 14),
        "systemmeasure": (203, 24, 29),
        "system_measure": (203, 24, 29),
    }
    normalized = name.lower().replace(" ", "_").replace("-", "_")
    return palette.get(normalized, (80, 80, 80))


def extract_detections(result: Any) -> list[dict[str, Any]]:
    names = result.names
    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is None:
        return detections

    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    for box, confidence, class_id in zip(xyxy, confs, classes):
        class_index = int(class_id)
        detections.append(
            {
                "class_id": class_index,
                "class_name": str(names.get(class_index, class_index)),
                "confidence": float(confidence),
                "bbox": [float(value) for value in box],
            }
        )
    return detections


def annotate_page(image_path: Path, detections: list[dict[str, Any]], out_path: Path) -> None:
    Image, ImageDraw, ImageFont = require_pillow()
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for detection in detections:
        name = detection["class_name"]
        confidence = detection["confidence"]
        left, top, right, bottom = detection["bbox"]
        color = detection_color(name)
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        label = f"{name} {confidence:.2f}"
        text_bbox = draw.textbbox((left, top), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((left, top), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows]
    class_totals: dict[str, int] = {}
    for row in rows:
        for class_name, count in row["counts"].items():
            class_totals[class_name] = class_totals.get(class_name, 0) + count

    return {
        "pages": len(rows),
        "detections": sum(sum(row["counts"].values()) for row in rows),
        "class_totals": dict(sorted(class_totals.items())),
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
        },
    }


def write_summary_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    classes = sorted({class_name for row in rows for class_name in row["counts"]})
    fieldnames = [
        "source",
        "page",
        "width",
        "height",
        "latency_ms",
        "detections",
        *classes,
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            counts = row["counts"]
            csv_row = {
                "source": row["source"],
                "page": row["page"],
                "width": row["width"],
                "height": row["height"],
                "latency_ms": f"{row['latency_ms']:.2f}",
                "detections": sum(counts.values()),
            }
            csv_row.update({class_name: counts.get(class_name, 0) for class_name in classes})
            writer.writerow(csv_row)


def main() -> int:
    args = parse_args()
    YOLO = require_ultralytics()
    Image, _, _ = require_pillow()

    files = iter_input_files(args.inputs)
    if not files:
        raise SystemExit("No supported image or PDF files found.")
    if any(file_path.suffix.lower() in PDF_EXTENSIONS for file_path in files):
        require_pdfium()

    if not args.model.exists():
        if args.download_model:
            download_model(args.model)
        else:
            raise SystemExit(f"Model not found: {args.model}")

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    with TemporaryDirectory(prefix="simple-ai-omr-eval-") as tmp:
        pages = collect_pages(files, Path(tmp), args.pdf_dpi, args.max_pages)
        if not pages:
            raise SystemExit("No pages to evaluate.")

        rows: list[dict[str, Any]] = []
        for index, page in enumerate(pages, start=1):
            print(
                f"[{index}/{len(pages)}] {page.source} page {page.page_index + 1}",
                file=sys.stderr,
            )
            with Image.open(page.image_path) as image:
                width, height = image.size

            start = time.perf_counter()
            result = model.predict(
                source=str(page.image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            latency_ms = (time.perf_counter() - start) * 1000.0
            detections = extract_detections(result)

            counts: dict[str, int] = {}
            for detection in detections:
                class_name = detection["class_name"]
                counts[class_name] = counts.get(class_name, 0) + 1

            row = {
                "source": str(page.source),
                "page": page.page_index + 1,
                "width": width,
                "height": height,
                "latency_ms": latency_ms,
                "counts": dict(sorted(counts.items())),
                "detections": detections,
            }
            rows.append(row)

            if args.annotate:
                output_name = f"{safe_stem(page.source)}-p{page.page_index + 1:04d}.jpg"
                annotate_page(page.image_path, detections, args.out / "annotated" / output_name)

    summary = summarize(rows)
    (args.out / "detections.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary_csv(rows, args.out / "summary.csv")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
