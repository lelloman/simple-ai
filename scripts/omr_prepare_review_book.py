#!/usr/bin/env python3
"""Prepare a review dataset for correcting OMR system boxes.

This script renders score PDFs/images, runs a YOLO OMR layout model, and writes
an editable review directory. The review data is intentionally separate from
YOLO training export so humans can correct boxes before anything is added to
the training set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

try:
    import evaluate_omr_layout as omr_io
    import omr_predict_staff_system_yolo as yolo_io
except ImportError as exc:
    raise SystemExit(
        "Run this script from the repository root or from the scripts directory."
    ) from exc


REVIEW_VERSION = 1
DEFAULT_REVIEW_CLASSES = ("system",)
DEFAULT_HELPER_CLASSES = ("grand_staff",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render books and seed editable OMR review boxes from a YOLO model.",
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf-dpi", type=int, default=300)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--review-classes",
        default=",".join(DEFAULT_REVIEW_CLASSES),
        help="Comma-separated classes to make editable. Default: system",
    )
    parser.add_argument(
        "--helper-classes",
        default=",".join(DEFAULT_HELPER_CLASSES),
        help="Comma-separated non-editable helper classes to keep. Default: grand_staff",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace the output directory if it already exists.",
    )
    return parser.parse_args()


def parse_class_list(raw: str) -> list[str]:
    classes = [name.strip() for name in raw.split(",") if name.strip()]
    if not classes:
        raise SystemExit("class lists must contain at least one class")
    if len(set(classes)) != len(classes):
        raise SystemExit(f"class list contains duplicates: {raw}")
    return classes


def box_id(page_number: int, index: int) -> str:
    return f"p{page_number:04d}-box-{index:04d}"


def copy_page_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def detection_to_review_box(
    detection: dict[str, Any],
    page_number: int,
    index: int,
) -> dict[str, Any]:
    return {
        "id": box_id(page_number, index),
        "class": detection["class_name"],
        "bbox": detection["bbox"],
        "source": "model",
        "confidence": detection["confidence"],
        "status": "pending",
    }


def derived_region_to_review_box(
    region: yolo_io.OwnershipRegion,
    page_number: int,
    index: int,
) -> dict[str, Any]:
    return {
        "id": box_id(page_number, index),
        "class": "system",
        "bbox": region.region_bbox,
        "source": "derived-grand-staff",
        "confidence": region.confidence,
        "status": "pending",
        "derived_from": {
            "class": "grand_staff",
            "bbox": region.grand_staff_bbox,
        },
    }


def detection_to_helper_box(detection: dict[str, Any], page_number: int, index: int) -> dict[str, Any]:
    return {
        "id": f"p{page_number:04d}-helper-{index:04d}",
        "class": detection["class_name"],
        "bbox": detection["bbox"],
        "source": "model",
        "confidence": detection["confidence"],
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.start_page < 1:
        raise SystemExit("--start-page must be at least 1")
    if args.pdf_dpi <= 0:
        raise SystemExit("--pdf-dpi must be positive")
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"Output already exists: {args.out}. Pass --force to replace it.")
        shutil.rmtree(args.out)
    if not args.weights.exists():
        raise SystemExit(f"Missing weights: {args.weights}")

    review_classes = parse_class_list(args.review_classes)
    helper_classes = parse_class_list(args.helper_classes)
    class_filter = set(review_classes) | set(helper_classes)
    files = omr_io.iter_input_files(args.inputs)
    if not files:
        raise SystemExit("No supported input files found.")
    if any(file_path.suffix.lower() in omr_io.PDF_EXTENSIONS for file_path in files):
        omr_io.require_pdfium()
    YOLO = yolo_io.require_yolo()
    Image, _, _ = omr_io.require_pillow()

    args.out.mkdir(parents=True, exist_ok=True)
    pages_dir = args.out / "pages"
    predictions_dir = args.out / "predictions"
    reviewed_dir = args.out / "reviewed"
    model = YOLO(str(args.weights))
    manifest_pages: list[dict[str, Any]] = []
    total_review_boxes = 0
    total_helper_boxes = 0

    pages = yolo_io.collect_selected_pages(
        files,
        args.out / ".rendered",
        args.pdf_dpi,
        args.start_page,
        args.max_pages,
    )
    for index, page in enumerate(pages, start=1):
        page_number = page.page_index + 1
        print(f"[{index}/{len(pages)}] {page.source} page {page_number}", file=sys.stderr)
        page_stem = f"page-{index:04d}"
        page_image = pages_dir / f"{page_stem}.png"
        copy_page_image(page.image_path, page_image)
        with Image.open(page_image) as image:
            width, height = image.size

        start = time.perf_counter()
        result = model.predict(
            source=str(page_image),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        detections = yolo_io.extract_detections(result)
        detections = [
            detection
            for detection in detections
            if detection["class_name"] in class_filter
        ]
        latency_ms = (time.perf_counter() - start) * 1000.0
        review_boxes = [
            detection_to_review_box(detection, page_number, box_index)
            for box_index, detection in enumerate(
                [item for item in detections if item["class_name"] in review_classes],
                start=1,
            )
        ]
        if "system" in review_classes and not review_boxes:
            review_boxes = [
                derived_region_to_review_box(region, page_number, box_index)
                for box_index, region in enumerate(
                    yolo_io.compute_grand_staff_ownership_regions(detections, width, height),
                    start=1,
                )
            ]
        helper_boxes = [
            detection_to_helper_box(detection, page_number, box_index)
            for box_index, detection in enumerate(
                [item for item in detections if item["class_name"] in helper_classes],
                start=1,
            )
        ]
        total_review_boxes += len(review_boxes)
        total_helper_boxes += len(helper_boxes)

        page_data = {
            "version": REVIEW_VERSION,
            "page": page_number,
            "sequence": index,
            "source": str(page.source),
            "image": relative(page_image, args.out),
            "width": width,
            "height": height,
            "status": "pending",
            "boxes": review_boxes,
            "helpers": helper_boxes,
        }
        prediction_data = {
            **page_data,
            "latency_ms": latency_ms,
            "raw_detections": detections,
        }
        prediction_path = predictions_dir / f"{page_stem}.json"
        review_path = reviewed_dir / f"{page_stem}.json"
        write_json(prediction_path, prediction_data)
        write_json(review_path, page_data)
        manifest_pages.append(
            {
                "sequence": index,
                "page": page_number,
                "source": str(page.source),
                "image": relative(page_image, args.out),
                "prediction": relative(prediction_path, args.out),
                "review": relative(review_path, args.out),
                "status": "pending",
            }
        )

    shutil.rmtree(args.out / ".rendered", ignore_errors=True)
    manifest = {
        "version": REVIEW_VERSION,
        "id": str(uuid.uuid4()),
        "source_inputs": [str(path) for path in files],
        "render_dpi": args.pdf_dpi,
        "model": {
            "path": str(args.weights),
            "confidence": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
        },
        "review_classes": review_classes,
        "helper_classes": helper_classes,
        "pages": manifest_pages,
    }
    write_json(args.out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "review_dir": str(args.out),
                "pages": len(manifest_pages),
                "review_boxes": total_review_boxes,
                "helper_boxes": total_helper_boxes,
                "manifest": str(args.out / "manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
