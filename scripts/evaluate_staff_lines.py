#!/usr/bin/env python3
"""Evaluate classical staff-line based music score region detection.

This does not use ML. It rasterizes PDF pages, finds dense horizontal ink rows,
groups them into five-line staves, and groups nearby staves into systems.

It is intended as a quick sanity check against the YOLO OMR layout model:

    scripts/evaluate_staff_lines.py score.pdf --out /tmp/staff-eval --max-pages 5

Outputs:
    detections.json   Per-page staves/systems.
    summary.csv       Per-page counts.
    annotated/        Full-page images with staff/system boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
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
class Staff:
    bbox: tuple[int, int, int, int]
    line_rows: tuple[int, int, int, int, int]
    line_spacing: float
    confidence: float


@dataclass(frozen=True)
class System:
    bbox: tuple[int, int, int, int]
    staff_indices: tuple[int, ...]
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classical staff-line detection on score PDFs/images.",
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("staff-line-eval"))
    parser.add_argument("--pdf-dpi", type=int, default=300)
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="First 1-based PDF page to process. Default: 1",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument(
        "--ink-threshold",
        type=int,
        default=190,
        help="Pixels darker than this are treated as ink. Default: 190",
    )
    parser.add_argument(
        "--row-density",
        type=float,
        default=0.45,
        help="Minimum row darkness density to consider a staff-line row. Default: 0.45",
    )
    parser.add_argument(
        "--min-line-width",
        type=float,
        default=0.22,
        help="Minimum horizontal coverage ratio for a staff line. Default: 0.22",
    )
    parser.add_argument(
        "--staff-padding",
        type=float,
        default=1.6,
        help="Vertical padding around detected five-line staves, in staff spaces. Default: 1.6",
    )
    parser.add_argument(
        "--system-gap",
        type=float,
        default=4.2,
        help="Max vertical staff gap, in staff spaces, to group staves into one system. Default: 4.2",
    )
    parser.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def require_pillow() -> tuple[Any, Any, Any, Any]:
    Image, ImageDraw, ImageFont = omr_io.require_pillow()
    try:
        from PIL import ImageOps
    except ImportError as exc:
        raise SystemExit("Missing pillow ImageOps") from exc
    return Image, ImageDraw, ImageFont, ImageOps


def dark_row_segments(
    image: Any,
    ink_threshold: int,
    row_density: float,
) -> list[tuple[int, int, int]]:
    """Return dense horizontal row segments as (top, bottom, dark_count)."""
    _Image, _, _, ImageOps = require_pillow()
    gray = ImageOps.grayscale(image)
    width, height = gray.size
    pixels = gray.load()
    candidate_rows: list[tuple[int, int]] = []

    for y in range(height):
        darkness = 0
        for x in range(width):
            darkness += max(0, 255 - pixels[x, y])
        darkness_density = darkness / (255 * width)
        if darkness_density >= row_density:
            candidate_rows.append((y, round(darkness_density * width)))

    if not candidate_rows:
        return []

    segments: list[tuple[int, int, int]] = []
    start_y, total_dark = candidate_rows[0]
    prev_y = start_y
    count = 1
    for y, dark_count in candidate_rows[1:]:
        if y <= prev_y + 1:
            prev_y = y
            total_dark += dark_count
            count += 1
            continue
        segments.append((start_y, prev_y, round(total_dark / count)))
        start_y = y
        prev_y = y
        total_dark = dark_count
        count = 1
    segments.append((start_y, prev_y, round(total_dark / count)))
    return segments


def merge_close_rows(segments: list[tuple[int, int, int]], max_gap: int = 2) -> list[tuple[int, int, int]]:
    if not segments:
        return []
    merged: list[tuple[int, int, int]] = []
    start, end, score = segments[0]
    scores = [score]
    for next_start, next_end, next_score in segments[1:]:
        if next_start <= end + max_gap + 1:
            end = next_end
            scores.append(next_score)
        else:
            merged.append((start, end, round(statistics.fmean(scores))))
            start, end, scores = next_start, next_end, [next_score]
    merged.append((start, end, round(statistics.fmean(scores))))
    return merged


def row_centers(segments: list[tuple[int, int, int]]) -> list[tuple[float, int]]:
    return [((top + bottom) / 2.0, score) for top, bottom, score in segments]


def find_nearest_row(
    centers: list[tuple[float, int]],
    target: float,
    min_index: int,
    tolerance: float,
) -> int | None:
    best_index: int | None = None
    best_distance: float | None = None
    for index in range(min_index, len(centers)):
        distance = abs(centers[index][0] - target)
        if distance > tolerance and centers[index][0] > target:
            break
        if distance <= tolerance and (best_distance is None or distance < best_distance):
            best_index = index
            best_distance = distance
    return best_index


def candidate_staff_lines(
    centers: list[tuple[float, int]],
    min_spacing: float = 5.0,
    max_spacing: float = 55.0,
) -> list[tuple[list[int], float, float]]:
    """Find repeated five-line patterns without assuming rows are consecutive."""
    candidates: list[tuple[list[int], float, float]] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    for first_index, first in enumerate(centers):
        for second_index in range(first_index + 1, len(centers)):
            spacing = centers[second_index][0] - first[0]
            if spacing < min_spacing:
                continue
            if spacing > max_spacing:
                break

            tolerance = max(2.5, spacing * 0.18)
            line_indices = [first_index, second_index]
            previous_index = second_index
            miss = False
            for offset in range(2, 5):
                target = first[0] + offset * spacing
                match_index = find_nearest_row(
                    centers,
                    target,
                    previous_index + 1,
                    tolerance,
                )
                if match_index is None:
                    miss = True
                    break
                line_indices.append(match_index)
                previous_index = match_index

            if miss:
                continue

            key = tuple(line_indices)
            if key in seen:
                continue
            seen.add(key)
            rows = [centers[index][0] for index in line_indices]
            gaps = [rows[index + 1] - rows[index] for index in range(4)]
            local_spacing = statistics.fmean(gaps)
            regularity_error = statistics.fmean(abs(gap - local_spacing) for gap in gaps)
            regularity = max(0.0, 1.0 - regularity_error / max(local_spacing, 1.0))
            ink_score = statistics.fmean(centers[index][1] for index in line_indices)
            candidates.append((line_indices, local_spacing, regularity * ink_score))

    return sorted(candidates, key=lambda item: item[2], reverse=True)


def estimate_staff_x_bounds(
    image: Any,
    line_rows: tuple[int, int, int, int, int],
    ink_threshold: int,
    min_line_width: float,
) -> tuple[int, int] | None:
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    columns = [0] * width
    row_window = 1

    for row in line_rows:
        for y in range(max(0, row - row_window), min(height, row + row_window + 1)):
            for x in range(width):
                if pixels[x, y] < ink_threshold:
                    columns[x] += 1

    threshold = max(2, len(line_rows) // 2)
    candidate_columns = [index for index, value in enumerate(columns) if value >= threshold]
    if not candidate_columns:
        return None

    left = min(candidate_columns)
    right = max(candidate_columns)
    if (right - left + 1) / width < min_line_width:
        return None
    return left, right


def detect_staves(
    image: Any,
    ink_threshold: int,
    row_density: float,
    min_line_width: float,
    staff_padding: float,
) -> list[Staff]:
    width, height = image.size
    segments = merge_close_rows(dark_row_segments(image, ink_threshold, row_density))
    centers = row_centers(segments)

    staves: list[Staff] = []
    used_rows: set[int] = set()
    used_bands: list[tuple[int, int]] = []
    for line_indices, local_spacing, candidate_score in candidate_staff_lines(centers):
        if any(row_index in used_rows for row_index in line_indices):
            continue

        lines = [centers[row_index] for row_index in line_indices]
        rounded_rows = tuple(round(line[0]) for line in lines)
        x_bounds = estimate_staff_x_bounds(
            image,
            rounded_rows,
            ink_threshold,
            min_line_width,
        )
        if x_bounds is None:
            continue

        pad = round(local_spacing * staff_padding)
        left, right = x_bounds
        top = max(0, round(lines[0][0]) - pad)
        bottom = min(height - 1, round(lines[4][0]) + pad)
        if any(not (bottom < band_top or top > band_bottom) for band_top, band_bottom in used_bands):
            continue

        score = (candidate_score / width) / max(row_density, 0.01)
        confidence = max(0.0, min(1.0, score))
        staves.append(
            Staff(
                bbox=(left, top, right, bottom),
                line_rows=rounded_rows,
                line_spacing=local_spacing,
                confidence=confidence,
            )
        )
        used_rows.update(line_indices)
        used_bands.append((top, bottom))

    return sorted(staves, key=lambda staff: staff.bbox[1])


def group_systems(staves: list[Staff], system_gap: float) -> list[System]:
    if not staves:
        return []
    indexed = sorted(enumerate(staves), key=lambda item: item[1].bbox[1])
    groups: list[list[tuple[int, Staff]]] = [[indexed[0]]]

    for item in indexed[1:]:
        _, staff = item
        previous = groups[-1][-1][1]
        previous_gap = staff.bbox[1] - previous.bbox[3]
        spacing = statistics.fmean([previous.line_spacing, staff.line_spacing])
        if previous_gap <= spacing * system_gap:
            groups[-1].append(item)
        else:
            groups.append([item])

    systems: list[System] = []
    for group in groups:
        left = min(staff.bbox[0] for _, staff in group)
        top = min(staff.bbox[1] for _, staff in group)
        right = max(staff.bbox[2] for _, staff in group)
        bottom = max(staff.bbox[3] for _, staff in group)
        confidence = statistics.fmean(staff.confidence for _, staff in group)
        systems.append(
            System(
                bbox=(left, top, right, bottom),
                staff_indices=tuple(index for index, _staff in group),
                confidence=confidence,
            )
        )
    return systems


def detections_for_page(staves: list[Staff], systems: list[System]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for system in systems:
        detections.append(
            {
                "class_name": "system",
                "confidence": system.confidence,
                "bbox": list(system.bbox),
                "staff_indices": list(system.staff_indices),
            }
        )
    for index, staff in enumerate(staves):
        detections.append(
            {
                "class_name": "staff",
                "confidence": staff.confidence,
                "bbox": list(staff.bbox),
                "line_rows": list(staff.line_rows),
                "staff_index": index,
            }
        )
    return detections


def annotate_page(image_path: Path, staves: list[Staff], systems: list[System], output_path: Path) -> None:
    Image, ImageDraw, ImageFont, _ = require_pillow()
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for system_index, system in enumerate(systems, start=1):
        left, top, right, bottom = system.bbox
        draw.rectangle((left, top, right, bottom), outline=(217, 95, 14), width=5)
        label = f"system {system_index}"
        text_bbox = draw.textbbox((left, top), label, font=font)
        draw.rectangle(text_bbox, fill=(217, 95, 14))
        draw.text((left, top), label, fill=(255, 255, 255), font=font)

    for staff_index, staff in enumerate(staves, start=1):
        left, top, right, bottom = staff.bbox
        draw.rectangle((left, top, right, bottom), outline=(35, 139, 69), width=3)
        label = f"staff {staff_index}"
        text_bbox = draw.textbbox((left, bottom - 12), label, font=font)
        draw.rectangle(text_bbox, fill=(35, 139, 69))
        draw.text((left, bottom - 12), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "page", "width", "height", "staves", "systems"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source": row["source"],
                    "page": row["page"],
                    "width": row["width"],
                    "height": row["height"],
                    "staves": row["counts"]["staff"],
                    "systems": row["counts"]["system"],
                }
            )


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
        elif suffix in omr_io.PDF_EXTENSIONS:
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


def main() -> int:
    args = parse_args()
    Image, _, _, _ = require_pillow()
    files = omr_io.iter_input_files(args.inputs)
    if not files:
        raise SystemExit("No supported image or PDF files found.")
    if any(file_path.suffix.lower() in omr_io.PDF_EXTENSIONS for file_path in files):
        omr_io.require_pdfium()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="simple-ai-staff-eval-") as tmp:
        pages = collect_selected_pages(
            files,
            Path(tmp),
            args.pdf_dpi,
            args.start_page,
            args.max_pages,
        )
        for page_number, page in enumerate(pages, start=1):
            print(f"[{page_number}/{len(pages)}] {page.source} page {page.page_index + 1}", file=sys.stderr)
            image = Image.open(page.image_path).convert("RGB")
            staves = detect_staves(
                image,
                args.ink_threshold,
                args.row_density,
                args.min_line_width,
                args.staff_padding,
            )
            systems = group_systems(staves, args.system_gap)
            detections = detections_for_page(staves, systems)

            row = {
                "source": str(page.source),
                "page": page.page_index + 1,
                "width": image.width,
                "height": image.height,
                "counts": {
                    "staff": len(staves),
                    "system": len(systems),
                },
                "detections": detections,
            }
            rows.append(row)

            if args.annotate:
                output_name = f"{omr_io.safe_stem(page.source)}-p{page.page_index + 1:04d}.jpg"
                annotate_page(page.image_path, staves, systems, args.out / "annotated" / output_name)

    (args.out / "detections.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary_csv(rows, args.out / "summary.csv")
    summary = {
        "pages": len(rows),
        "staves": sum(row["counts"]["staff"] for row in rows),
        "systems": sum(row["counts"]["system"] for row in rows),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
