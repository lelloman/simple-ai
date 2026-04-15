#!/usr/bin/env python3
"""Detect music system regions on score pages using classical image processing.

This is a standalone evaluator for the actual feature we care about:
enumerating system bounding boxes on scanned/printed score pages.

It does not transcribe music and does not detect symbols. It extracts long
horizontal staff-line components, groups them into staves, then groups staves
into systems using adaptive page geometry.

Run:

    . .venv-omr/bin/activate
    pip install -r scripts/omr-layout-requirements.txt
    scripts/evaluate_system_regions.py score.pdf --out /tmp/system-eval --start-page 10 --max-pages 5

Outputs:
    systems.csv        One row per detected system.
    detections.json    Page/staff/system details for inspection.
    annotated/         Full-page images with numbered system boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

try:
    import evaluate_omr_layout as omr_io
except ImportError as exc:
    raise SystemExit(
        "Run this script from the repository root or from the scripts directory."
    ) from exc


@dataclass(frozen=True)
class LineSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int


@dataclass(frozen=True)
class StaffRegion:
    bbox: tuple[int, int, int, int]
    line_y: tuple[int, int, int, int, int]
    spacing: float
    confidence: float


@dataclass(frozen=True)
class SystemRegion:
    bbox: tuple[int, int, int, int]
    staff_indices: tuple[int, ...]
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and enumerate music system regions on score pages.",
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("system-region-eval"))
    parser.add_argument("--pdf-dpi", type=int, default=300)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument(
        "--min-line-width",
        type=float,
        default=0.28,
        help="Minimum staff-line width as page-width ratio. Default: 0.28",
    )
    parser.add_argument(
        "--horizontal-kernel",
        type=float,
        default=0.035,
        help="Horizontal morphology kernel as page-width ratio. Default: 0.035",
    )
    parser.add_argument(
        "--min-spacing",
        type=float,
        default=5.0,
        help="Minimum staff-line spacing in pixels. Default: 5",
    )
    parser.add_argument(
        "--max-spacing",
        type=float,
        default=70.0,
        help="Maximum staff-line spacing in pixels. Default: 70",
    )
    parser.add_argument(
        "--staff-padding",
        type=float,
        default=1.5,
        help="Padding around each staff in staff spaces. Default: 1.5",
    )
    parser.add_argument(
        "--system-padding",
        type=float,
        default=1.1,
        help="Padding around each system in median staff spaces. Default: 1.1",
    )
    parser.add_argument(
        "--split-gap-ratio",
        type=float,
        default=2.4,
        help="Split systems when staff gap exceeds this many staff spaces. Default: 2.4",
    )
    parser.add_argument(
        "--show-staves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw staff boxes in annotated images. Default: false",
    )
    parser.add_argument(
        "--annotated-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one annotated PDF containing all processed pages. Default: true",
    )
    return parser.parse_args()


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: opencv-python-headless. Install with: "
            "pip install -r scripts/omr-layout-requirements.txt"
        ) from exc
    return cv2


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: numpy. It is installed with opencv-python-headless."
        ) from exc
    return np


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
        page_count = len(document)
        first_index = start_page - 1
        if first_index >= page_count:
            continue
        last_index = page_count
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


def extract_horizontal_lines(
    image_path: Path,
    min_line_width_ratio: float,
    horizontal_kernel_ratio: float,
) -> list[LineSegment]:
    cv2 = require_cv2()
    np = require_numpy()

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = gray.shape
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )

    kernel_width = max(25, int(width * horizontal_kernel_ratio))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(horizontal, 8)
    min_width = width * min_line_width_ratio
    segments: list[LineSegment] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if w < min_width:
            continue
        if h > max(12, width * 0.01):
            continue
        if area < w * 0.35:
            continue

        component_mask = labels == label
        ys, xs = np.where(component_mask)
        if len(xs) == 0:
            continue
        segments.append(
            LineSegment(
                x1=int(xs.min()),
                y1=int(round(float(centroids[label][1]))),
                x2=int(xs.max()),
                y2=int(round(float(centroids[label][1]))),
                width=w,
                height=h,
            )
        )

    return merge_same_line_segments(sorted(segments, key=lambda line: (line.y1, line.x1)))


def merge_same_line_segments(lines: list[LineSegment]) -> list[LineSegment]:
    if not lines:
        return []

    merged: list[LineSegment] = []
    group: list[LineSegment] = [lines[0]]
    for line in lines[1:]:
        current_y = statistics.median(member.y1 for member in group)
        if abs(line.y1 - current_y) <= 3:
            group.append(line)
            continue
        merged.append(merge_line_group(group))
        group = [line]
    merged.append(merge_line_group(group))
    return merged


def merge_line_group(group: list[LineSegment]) -> LineSegment:
    x1 = min(line.x1 for line in group)
    x2 = max(line.x2 for line in group)
    y = round(statistics.fmean(line.y1 for line in group))
    height = max(line.height for line in group)
    return LineSegment(x1=x1, y1=y, x2=x2, y2=y, width=x2 - x1 + 1, height=height)


def staff_candidate_score(lines: list[LineSegment], spacing: float) -> float:
    ys = [line.y1 for line in lines]
    gaps = [ys[index + 1] - ys[index] for index in range(4)]
    regularity_error = statistics.fmean(abs(gap - spacing) for gap in gaps)
    regularity = max(0.0, 1.0 - regularity_error / max(spacing, 1.0))

    left = max(line.x1 for line in lines)
    right = min(line.x2 for line in lines)
    overlap = max(0, right - left + 1)
    widest = max(line.width for line in lines)
    overlap_ratio = overlap / max(widest, 1)
    width_consistency = min(line.width for line in lines) / max(widest, 1)
    return regularity * 0.5 + overlap_ratio * 0.3 + width_consistency * 0.2


def detect_staves(
    lines: list[LineSegment],
    page_width: int,
    page_height: int,
    min_spacing: float,
    max_spacing: float,
    staff_padding: float,
) -> list[StaffRegion]:
    candidates: list[tuple[float, tuple[int, int, int, int, int], float]] = []
    for start in range(len(lines) - 4):
        for second in range(start + 1, len(lines) - 3):
            spacing = lines[second].y1 - lines[start].y1
            if spacing < min_spacing:
                continue
            if spacing > max_spacing:
                break

            indices = [start, second]
            previous = second
            ok = True
            for offset in range(2, 5):
                target = lines[start].y1 + offset * spacing
                tolerance = max(3.0, spacing * 0.18)
                match = None
                for candidate in range(previous + 1, len(lines)):
                    if lines[candidate].y1 > target + tolerance:
                        break
                    if abs(lines[candidate].y1 - target) <= tolerance:
                        match = candidate
                        break
                if match is None:
                    ok = False
                    break
                indices.append(match)
                previous = match

            if not ok:
                continue
            candidate_lines = [lines[index] for index in indices]
            score = staff_candidate_score(candidate_lines, spacing)
            if score < 0.62:
                continue
            candidates.append((score, tuple(indices), spacing))

    candidates.sort(reverse=True, key=lambda item: item[0])
    used_lines: set[int] = set()
    used_bands: list[tuple[int, int]] = []
    staves: list[StaffRegion] = []
    for score, indices, spacing in candidates:
        if any(index in used_lines for index in indices):
            continue
        candidate_lines = [lines[index] for index in indices]
        left = min(line.x1 for line in candidate_lines)
        right = max(line.x2 for line in candidate_lines)
        if (right - left + 1) / page_width < 0.22:
            continue
        pad = round(spacing * staff_padding)
        top = max(0, candidate_lines[0].y1 - pad)
        bottom = min(page_height - 1, candidate_lines[-1].y1 + pad)
        if any(not (bottom < band_top or top > band_bottom) for band_top, band_bottom in used_bands):
            continue

        used_lines.update(indices)
        used_bands.append((top, bottom))
        staves.append(
            StaffRegion(
                bbox=(left, top, right, bottom),
                line_y=tuple(line.y1 for line in candidate_lines),  # type: ignore[arg-type]
                spacing=spacing,
                confidence=max(0.0, min(1.0, score)),
            )
        )

    return sorted(staves, key=lambda staff: (staff.bbox[1], staff.bbox[0]))


def adaptive_system_split_gap(staves: list[StaffRegion], split_gap_ratio: float) -> float:
    if not staves:
        return 0.0
    median_spacing = statistics.median(staff.spacing for staff in staves)
    gaps = [
        staves[index + 1].bbox[1] - staves[index].bbox[3]
        for index in range(len(staves) - 1)
    ]
    positive_gaps = [gap for gap in gaps if gap > 0]
    if not positive_gaps:
        return median_spacing * split_gap_ratio

    ratio_threshold = median_spacing * split_gap_ratio
    if len(positive_gaps) < 3:
        return ratio_threshold

    sorted_gaps = sorted(positive_gaps)
    largest_jump = 0.0
    jump_threshold = ratio_threshold
    for left, right in zip(sorted_gaps, sorted_gaps[1:]):
        jump = right - left
        if jump > largest_jump:
            largest_jump = jump
            jump_threshold = (left + right) / 2.0

    return max(ratio_threshold, jump_threshold)


def group_systems(
    staves: list[StaffRegion],
    page_width: int,
    page_height: int,
    system_padding: float,
    split_gap_ratio: float,
) -> list[SystemRegion]:
    if not staves:
        return []

    staves = sorted(staves, key=lambda staff: staff.bbox[1])
    split_gap = adaptive_system_split_gap(staves, split_gap_ratio)
    groups: list[list[int]] = [[0]]
    for index in range(1, len(staves)):
        previous = staves[index - 1]
        current = staves[index]
        vertical_gap = current.bbox[1] - previous.bbox[3]
        horizontal_overlap = min(previous.bbox[2], current.bbox[2]) - max(previous.bbox[0], current.bbox[0])
        overlap_ratio = horizontal_overlap / max(
            min(previous.bbox[2] - previous.bbox[0], current.bbox[2] - current.bbox[0]),
            1,
        )
        if vertical_gap > split_gap or overlap_ratio < 0.45:
            groups.append([index])
        else:
            groups[-1].append(index)

    median_spacing = statistics.median(staff.spacing for staff in staves)
    pad = round(median_spacing * system_padding)
    systems: list[SystemRegion] = []
    for group in groups:
        left = min(staves[index].bbox[0] for index in group)
        top = max(0, min(staves[index].bbox[1] for index in group) - pad)
        right = max(staves[index].bbox[2] for index in group)
        bottom = min(page_height - 1, max(staves[index].bbox[3] for index in group) + pad)
        confidence = statistics.fmean(staves[index].confidence for index in group)
        systems.append(
            SystemRegion(
                bbox=(left, top, right, bottom),
                staff_indices=tuple(group),
                confidence=confidence,
            )
        )
    return systems


def annotate(
    image_path: Path,
    output_path: Path,
    systems: list[SystemRegion],
    staves: list[StaffRegion],
    show_staves: bool,
) -> None:
    Image, ImageDraw, ImageFont = omr_io.require_pillow()
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if show_staves:
        for staff_index, staff in enumerate(staves, start=1):
            left, top, right, bottom = staff.bbox
            draw.rectangle((left, top, right, bottom), outline=(35, 139, 69), width=2)
            label = f"staff {staff_index}"
            text_bbox = draw.textbbox((left, bottom - 14), label, font=font)
            draw.rectangle(text_bbox, fill=(35, 139, 69))
            draw.text((left, bottom - 14), label, fill=(255, 255, 255), font=font)

    for system_index, system in enumerate(systems, start=1):
        left, top, right, bottom = system.bbox
        draw.rectangle((left, top, right, bottom), outline=(217, 95, 14), width=6)
        label = f"system {system_index}"
        text_bbox = draw.textbbox((left, top), label, font=font)
        draw.rectangle(text_bbox, fill=(217, 95, 14))
        draw.text((left, top), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_annotated_pdf(image_paths: list[Path], output_path: Path) -> None:
    if not image_paths:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    for image in rest:
        image.close()
    first.close()


def page_result(
    source: Path,
    page_index: int,
    page_width: int,
    page_height: int,
    staves: list[StaffRegion],
    systems: list[SystemRegion],
) -> dict[str, Any]:
    return {
        "source": str(source),
        "page": page_index + 1,
        "width": page_width,
        "height": page_height,
        "systems": [
            {
                "index": index,
                "bbox": list(system.bbox),
                "staff_indices": list(system.staff_indices),
                "confidence": system.confidence,
            }
            for index, system in enumerate(systems, start=1)
        ],
        "staves": [
            {
                "index": index,
                "bbox": list(staff.bbox),
                "line_y": list(staff.line_y),
                "spacing": staff.spacing,
                "confidence": staff.confidence,
            }
            for index, staff in enumerate(staves, start=1)
        ],
    }


def write_systems_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "page",
                "system",
                "left",
                "top",
                "right",
                "bottom",
                "staff_count",
                "confidence",
            ],
        )
        writer.writeheader()
        for row in rows:
            for system in row["systems"]:
                left, top, right, bottom = system["bbox"]
                writer.writerow(
                    {
                        "source": row["source"],
                        "page": row["page"],
                        "system": system["index"],
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "staff_count": len(system["staff_indices"]),
                        "confidence": f"{system['confidence']:.3f}",
                    }
                )


def main() -> int:
    args = parse_args()
    require_cv2()
    require_numpy()
    Image, _, _ = omr_io.require_pillow()

    files = omr_io.iter_input_files(args.inputs)
    if not files:
        raise SystemExit("No supported image or PDF files found.")
    if any(file_path.suffix.lower() in omr_io.PDF_EXTENSIONS for file_path in files):
        omr_io.require_pdfium()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    annotated_paths: list[Path] = []
    with TemporaryDirectory(prefix="simple-ai-system-eval-") as tmp:
        pages = collect_selected_pages(
            files,
            Path(tmp),
            args.pdf_dpi,
            args.start_page,
            args.max_pages,
        )
        for page_number, page in enumerate(pages, start=1):
            print(f"[{page_number}/{len(pages)}] {page.source} page {page.page_index + 1}", file=sys.stderr)
            with Image.open(page.image_path) as image:
                width, height = image.size

            lines = extract_horizontal_lines(
                page.image_path,
                args.min_line_width,
                args.horizontal_kernel,
            )
            staves = detect_staves(
                lines,
                width,
                height,
                args.min_spacing,
                args.max_spacing,
                args.staff_padding,
            )
            systems = group_systems(
                staves,
                width,
                height,
                args.system_padding,
                args.split_gap_ratio,
            )
            rows.append(page_result(page.source, page.page_index, width, height, staves, systems))

            output_name = f"{omr_io.safe_stem(page.source)}-p{page.page_index + 1:04d}.jpg"
            annotated_path = args.out / "annotated" / output_name
            annotate(page.image_path, annotated_path, systems, staves, args.show_staves)
            annotated_paths.append(annotated_path)

    write_systems_csv(rows, args.out / "systems.csv")
    (args.out / "detections.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "pages": len(rows),
        "systems": sum(len(row["systems"]) for row in rows),
        "staves": sum(len(row["staves"]) for row in rows),
    }
    if args.annotated_pdf:
        pdf_path = args.out / "annotated.pdf"
        write_annotated_pdf(annotated_paths, pdf_path)
        summary["annotated_pdf"] = str(pdf_path)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
