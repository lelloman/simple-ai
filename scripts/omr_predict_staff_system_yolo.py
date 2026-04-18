#!/usr/bin/env python3
"""Run a trained staff+system YOLO model on score PDFs/images and annotate pages."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

try:
    import evaluate_omr_layout as omr_io
except ImportError as exc:
    raise SystemExit(
        "Run this script from the repository root or from the scripts directory."
    ) from exc


CLASS_COLORS = {
    "system_measure": (117, 112, 179),
    "staff_measure": (231, 41, 138),
    "staff": (35, 139, 69),
    "system": (217, 95, 14),
    "grand_staff": (27, 158, 119),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict staff/system boxes with YOLO.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("staff-system-yolo-predictions"))
    parser.add_argument("--pdf-dpi", type=int, default=300)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show-staves", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--show-classes",
        default=None,
        help=(
            "Comma-separated class names to draw. Overrides --show-staves. "
            "Default: system only, or system,staff when --show-staves is set."
        ),
    )
    parser.add_argument("--annotated-pdf", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_show_classes(raw: str | None, show_staves: bool) -> set[str]:
    if raw is not None:
        classes = {name.strip() for name in raw.split(",") if name.strip()}
        if not classes:
            raise SystemExit("--show-classes must include at least one class")
        return classes
    if show_staves:
        return {"system", "staff"}
    return {"system"}


def require_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install with: pip install -r scripts/omr-layout-requirements.txt"
        ) from exc
    return YOLO


def collect_selected_pages(
    files: list[Path],
    work_dir: Path,
    pdf_dpi: int,
    start_page: int,
    max_pages: int | None,
) -> list[omr_io.PageInput]:
    if start_page < 1:
        raise SystemExit("--start-page must be at least 1")
    pages: list[omr_io.PageInput] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix in omr_io.IMAGE_EXTENSIONS:
            if start_page == 1:
                pages.append(omr_io.PageInput(file_path, 0, file_path))
            continue
        if suffix not in omr_io.PDF_EXTENSIONS:
            continue

        pdfium = omr_io.require_pdfium()
        scale = pdf_dpi / 72.0
        document = pdfium.PdfDocument(str(file_path))
        first_index = start_page - 1
        last_index = len(document)
        if max_pages is not None:
            last_index = min(last_index, first_index + max_pages)
        pdf_dir = work_dir / omr_io.safe_stem(file_path)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        for page_index in range(first_index, last_index):
            page = document[page_index]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            image_path = pdf_dir / f"page-{page_index + 1:04d}.png"
            image.save(image_path)
            pages.append(omr_io.PageInput(file_path, page_index, image_path))
    return pages


def extract_detections(result: Any) -> list[dict[str, Any]]:
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return []
    detections: list[dict[str, Any]] = []
    for xyxy, conf, cls in zip(
        boxes.xyxy.cpu().tolist(),
        boxes.conf.cpu().tolist(),
        boxes.cls.cpu().tolist(),
    ):
        class_id = int(cls)
        detections.append(
            {
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id)),
                "confidence": float(conf),
                "bbox": [float(value) for value in xyxy],
            }
        )
    return detections


def annotate_page(
    image_path: Path,
    output_path: Path,
    detections: list[dict[str, Any]],
    show_classes: set[str],
) -> None:
    Image, ImageDraw, ImageFont = omr_io.require_pillow()
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    system_index = 0
    for detection in sorted(detections, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        class_name = detection["class_name"]
        if class_name not in show_classes:
            continue
        left, top, right, bottom = detection["bbox"]
        color = CLASS_COLORS.get(class_name, (80, 80, 80))
        width = 6 if class_name == "system" else 3 if class_name == "grand_staff" else 2
        draw.rectangle((left, top, right, bottom), outline=color, width=width)
        if class_name == "system":
            system_index += 1
            label = f"system {system_index} {detection['confidence']:.2f}"
        else:
            label = f"{class_name} {detection['confidence']:.2f}"
        text_bbox = draw.textbbox((left, top), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((left, top), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_annotated_pdf(image_paths: list[Path], output_path: Path) -> None:
    if not image_paths:
        return
    img2pdf = shutil.which("img2pdf")
    if img2pdf:
        subprocess.run(
            [img2pdf, *[str(path) for path in image_paths], "-o", str(output_path)],
            check=True,
        )
        return
    Image, _, _ = omr_io.require_pillow()
    first = Image.open(image_paths[0]).convert("RGB")
    rest = [Image.open(path).convert("RGB") for path in image_paths[1:]]
    first.save(output_path, save_all=True, append_images=rest)


def main() -> int:
    args = parse_args()
    YOLO = require_yolo()
    Image, _, _ = omr_io.require_pillow()
    files = omr_io.iter_input_files(args.inputs)
    if any(file_path.suffix.lower() in omr_io.PDF_EXTENSIONS for file_path in files):
        omr_io.require_pdfium()
    if not files:
        raise SystemExit("No supported input files found.")
    if not args.weights.exists():
        raise SystemExit(f"Missing weights: {args.weights}")
    show_classes = parse_show_classes(args.show_classes, args.show_staves)

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    rows: list[dict[str, Any]] = []
    annotated: list[Path] = []
    with TemporaryDirectory(prefix="simple-ai-yolo-predict-") as tmp:
        pages = collect_selected_pages(files, Path(tmp), args.pdf_dpi, args.start_page, args.max_pages)
        for index, page in enumerate(pages, start=1):
            print(f"[{index}/{len(pages)}] {page.source} page {page.page_index + 1}", file=sys.stderr)
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
            detections = extract_detections(result)
            rows.append(
                {
                    "source": str(page.source),
                    "page": page.page_index + 1,
                    "width": width,
                    "height": height,
                    "latency_ms": (time.perf_counter() - start) * 1000.0,
                    "detections": detections,
                }
            )
            image_name = f"{omr_io.safe_stem(page.source)}-p{page.page_index + 1:04d}.jpg"
            output_path = args.out / "annotated" / image_name
            annotate_page(page.image_path, output_path, detections, show_classes)
            annotated.append(output_path)

    (args.out / "detections.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.annotated_pdf:
        write_annotated_pdf(annotated, args.out / "annotated.pdf")
    print(
        json.dumps(
            {
                "pages": len(rows),
                "detections": sum(len(row["detections"]) for row in rows),
                "annotated_pdf": str(args.out / "annotated.pdf") if args.annotated_pdf else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
