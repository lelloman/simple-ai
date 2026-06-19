#!/usr/bin/env python3
"""Export reviewed OMR system boxes as a YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASS_NAMES = ["system"]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_PAGE_STATUSES = ("reviewed",)
DEFAULT_BOX_STATUSES = ("accepted",)


@dataclass(frozen=True)
class ExportExample:
    review_dir: Path
    sequence: int
    source_page: int
    image: Path
    review: Path
    width: int
    height: int
    labels: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reviewed OMR system boxes to a system-only YOLO dataset.",
    )
    parser.add_argument("review_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--page-statuses",
        default=",".join(DEFAULT_PAGE_STATUSES),
        help="Comma-separated page statuses to export. Default: reviewed",
    )
    parser.add_argument(
        "--box-statuses",
        default=",".join(DEFAULT_BOX_STATUSES),
        help="Comma-separated box statuses to export. Default: accepted",
    )
    parser.add_argument(
        "--keep-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export reviewed pages with no accepted boxes as negatives. Default: true",
    )
    parser.add_argument(
        "--copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy images instead of hard-linking. Default: true",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace output directory if it exists.",
    )
    return parser.parse_args()


def parse_statuses(raw: str, option_name: str) -> set[str]:
    statuses = {status.strip() for status in raw.split(",") if status.strip()}
    if not statuses:
        raise SystemExit(f"{option_name} must include at least one status")
    return statuses


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def bbox_to_yolo_line(bbox: list[float], image_width: int, image_height: int) -> str | None:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(bbox) != 4:
        return None

    left, top, right, bottom = [float(value) for value in bbox]
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    left = clamp(left, 0.0, float(image_width))
    right = clamp(right, 0.0, float(image_width))
    top = clamp(top, 0.0, float(image_height))
    bottom = clamp(bottom, 0.0, float(image_height))
    width = right - left
    height = bottom - top
    if width <= 0.0 or height <= 0.0:
        return None

    x_center = (left + right) / 2.0 / image_width
    y_center = (top + bottom) / 2.0 / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return f"0 {x_center:.8f} {y_center:.8f} {norm_width:.8f} {norm_height:.8f}"


def relative_to_review_dir(review_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return review_dir / path


def labels_from_review(
    review_data: dict[str, Any],
    box_statuses: set[str],
) -> list[str]:
    width = int(review_data["width"])
    height = int(review_data["height"])
    labels: list[str] = []
    for box in review_data.get("boxes", []):
        if box.get("class") != "system":
            continue
        if box.get("status") not in box_statuses:
            continue
        line = bbox_to_yolo_line(box.get("bbox", []), width, height)
        if line is not None:
            labels.append(line)
    return labels


def collect_examples(
    review_dirs: list[Path],
    page_statuses: set[str],
    box_statuses: set[str],
    keep_empty: bool,
) -> list[ExportExample]:
    examples: list[ExportExample] = []
    for review_dir in review_dirs:
        manifest_path = review_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"Missing manifest: {manifest_path}\n"
                "Run scripts/omr_prepare_review_book.py first, then pass its review_dir here."
            )
        manifest = read_json(manifest_path)
        for page in manifest.get("pages", []):
            if page.get("status") not in page_statuses:
                continue
            review_path = relative_to_review_dir(review_dir, page["review"])
            review_data = read_json(review_path)
            labels = labels_from_review(review_data, box_statuses)
            if not labels and not keep_empty:
                continue
            image_path = relative_to_review_dir(review_dir, review_data["image"])
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise SystemExit(f"Unsupported review image extension: {image_path}")
            if not image_path.exists():
                raise SystemExit(f"Missing review image: {image_path}")
            examples.append(
                ExportExample(
                    review_dir=review_dir,
                    sequence=int(review_data["sequence"]),
                    source_page=int(review_data["page"]),
                    image=image_path,
                    review=review_path,
                    width=int(review_data["width"]),
                    height=int(review_data["height"]),
                    labels=labels,
                )
            )
    return examples


def split_examples(
    examples: list[ExportExample],
    train_split: float,
    seed: int,
) -> dict[str, list[ExportExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        train_count = len(shuffled)
    else:
        train_count = round(len(shuffled) * train_split)
        train_count = max(1, min(train_count, len(shuffled) - 1))
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:],
    }


def stable_image_name(example: ExportExample) -> str:
    key = f"{example.review_dir.resolve()}:{example.sequence}:{example.image.resolve()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return (
        f"{example.review_dir.name}_p{example.source_page:04d}_"
        f"s{example.sequence:04d}_{digest}{example.image.suffix.lower()}"
    )


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def write_data_yaml(out_dir: Path) -> None:
    lines = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: system",
        "",
    ]
    (out_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def export_dataset(
    examples: list[ExportExample],
    out_dir: Path,
    train_split: float,
    seed: int,
    copy: bool,
) -> dict[str, Any]:
    partitions = split_examples(examples, train_split, seed)
    summary: dict[str, Any] = {
        "images": len(examples),
        "boxes": sum(len(example.labels) for example in examples),
        "partitions": {},
        "classes": CLASS_NAMES,
    }
    for partition, partition_examples in partitions.items():
        image_dir = out_dir / "images" / partition
        label_dir = out_dir / "labels" / partition
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        box_count = 0
        for example in partition_examples:
            image_name = stable_image_name(example)
            label_name = f"{Path(image_name).stem}.txt"
            link_or_copy(example.image, image_dir / image_name, copy)
            (label_dir / label_name).write_text(
                "\n".join(example.labels) + ("\n" if example.labels else ""),
                encoding="utf-8",
            )
            box_count += len(example.labels)
        summary["partitions"][partition] = {
            "images": len(partition_examples),
            "boxes": box_count,
        }
    write_data_yaml(out_dir)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    if not 0.0 < args.train_split < 1.0:
        raise SystemExit("--train-split must be between 0 and 1")
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"Output already exists: {args.out}. Pass --force to replace it.")
        shutil.rmtree(args.out)

    page_statuses = parse_statuses(args.page_statuses, "--page-statuses")
    box_statuses = parse_statuses(args.box_statuses, "--box-statuses")
    examples = collect_examples(args.review_dirs, page_statuses, box_statuses, args.keep_empty)
    if not examples:
        raise SystemExit("No reviewed examples matched the export filters.")

    summary = export_dataset(examples, args.out, args.train_split, args.seed, args.copy)
    print(json.dumps({**summary, "out": str(args.out), "data": str(args.out / "data.yaml")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
