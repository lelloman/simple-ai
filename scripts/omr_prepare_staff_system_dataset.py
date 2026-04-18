#!/usr/bin/env python3
"""Prepare a filtered YOLO dataset from OLA layout archives.

Input is one or more extracted upstream dataset directories, each with:

    images/
    labels/

The upstream class map is:

    0 system_measure
    1 staff_measure
    2 staff
    3 system
    4 grand_staff

By default this script keeps only:

    0 staff   <- upstream 2
    1 system  <- upstream 3

Use --classes to train richer layout detectors. For example:

    --classes staff,system,grand_staff,staff_measure,system_measure
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path("/tmp/simple-ai-omr-training")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
UPSTREAM_CLASSES = {
    "system_measure": "0",
    "staff_measure": "1",
    "staff": "2",
    "system": "3",
    "grand_staff": "4",
}
DEFAULT_CLASS_NAMES = ["staff", "system"]


@dataclass(frozen=True)
class Example:
    image: Path
    label: Path
    dataset: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter OLA YOLO labels to selected classes.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=None,
        help="Extracted dataset directory. Repeatable. Default: all extracted dirs under root.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ROOT / "datasets" / "staff-system-yolo",
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASS_NAMES),
        help=(
            "Comma-separated upstream class names to keep. Available: "
            f"{', '.join(UPSTREAM_CLASSES)}. Default: staff,system"
        ),
    )
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--copy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy images instead of hard-linking. Default: false",
    )
    parser.add_argument(
        "--keep-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep images that have no staff/system boxes as negatives. Default: true",
    )
    return parser.parse_args()


def parse_classes(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise SystemExit("--classes must select at least one class")
    unknown = [name for name in names if name not in UPSTREAM_CLASSES]
    if unknown:
        raise SystemExit(
            f"Unknown class(es): {', '.join(unknown)}. "
            f"Available: {', '.join(UPSTREAM_CLASSES)}"
        )
    if len(set(names)) != len(names):
        raise SystemExit("--classes contains duplicates")
    return names


def find_sources(root: Path, sources: list[Path] | None) -> list[Path]:
    if sources:
        return sources
    extracted = root / "extracted"
    return sorted(path for path in extracted.iterdir() if (path / "images").is_dir())


def iter_examples(source: Path) -> list[Example]:
    images_dir = source / "images"
    labels_dir = source / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Not a YOLO dataset: {source}")

    examples: list[Example] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = labels_dir / f"{image.stem}.txt"
        examples.append(Example(image=image, label=label, dataset=source.name))
    return examples


def remap_label(input_path: Path, class_remap: dict[str, str]) -> list[str]:
    if not input_path.exists():
        return []
    output_lines: list[str] = []
    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            continue
        mapped = class_remap.get(parts[0])
        if mapped is None:
            continue
        output_lines.append(" ".join([mapped, *parts[1:]]))
    return output_lines


def stable_name(example: Example) -> str:
    digest = hashlib.sha1(str(example.image).encode("utf-8")).hexdigest()[:10]
    return f"{example.dataset}_{example.image.stem}_{digest}{example.image.suffix.lower()}"


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def write_yaml(out_dir: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    lines.append("")
    yaml = "\n".join(lines)
    (out_dir / "data.yaml").write_text(yaml, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not 0.0 < args.train_split < 1.0:
        raise SystemExit("--train-split must be between 0 and 1")
    class_names = parse_classes(args.classes)
    class_remap = {
        UPSTREAM_CLASSES[name]: str(index) for index, name in enumerate(class_names)
    }

    sources = find_sources(args.root, args.source)
    if not sources:
        raise SystemExit("No extracted datasets found.")

    examples: list[tuple[Example, list[str]]] = []
    for source in sources:
        for example in iter_examples(source):
            labels = remap_label(example.label, class_remap)
            if labels or args.keep_empty:
                examples.append((example, labels))

    if not examples:
        raise SystemExit("No examples left after filtering.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split = round(len(examples) * args.train_split)
    partitions = {
        "train": examples[:split],
        "val": examples[split:],
    }

    if args.out.exists():
        shutil.rmtree(args.out)

    class_counts = {name: 0 for name in class_names}
    for partition, partition_examples in partitions.items():
        for example, labels in partition_examples:
            image_name = stable_name(example)
            label_name = f"{Path(image_name).stem}.txt"
            link_or_copy(
                example.image,
                args.out / "images" / partition / image_name,
                args.copy,
            )
            (args.out / "labels" / partition).mkdir(parents=True, exist_ok=True)
            (args.out / "labels" / partition / label_name).write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )
            for line in labels:
                class_counts[class_names[int(line.split()[0])]] += 1

    write_yaml(args.out, class_names)
    print(f"Dataset written: {args.out}")
    print(f"Images: {len(examples)}")
    print(f"Train: {len(partitions['train'])}")
    print(f"Val: {len(partitions['val'])}")
    print(f"Boxes: {class_counts}")
    print(f"YOLO data config: {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
