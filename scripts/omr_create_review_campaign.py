#!/usr/bin/env python3
"""Create a structured OMR review campaign from score and negative-page sources."""

from __future__ import annotations

import argparse
import json
import random
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import evaluate_omr_layout as omr_io
except ImportError as exc:
    raise SystemExit(
        "Run this script from the repository root or from the scripts directory."
    ) from exc


CAMPAIGN_VERSION = 1
DEFAULT_NAME = "systems-active-learning"


@dataclass(frozen=True)
class BookInfo:
    path: Path
    category: str
    page_count: int


@dataclass(frozen=True)
class ReviewBatch:
    index: int
    category: str
    source: Path
    start_page: int
    max_pages: int
    review_dir: Path

    @property
    def page_count(self) -> int:
        return self.max_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, and optionally prepare, a manifest-backed OMR review campaign. "
            "Use score inputs for positive pages and --negative-input for "
            "text-only, cover, newspaper, or other non-music layouts."
        ),
    )
    parser.add_argument("score_inputs", nargs="+", type=Path)
    parser.add_argument(
        "--negative-input",
        action="append",
        default=[],
        type=Path,
        help="Path to non-score PDFs/images. Can be passed multiple times.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Optional upstream/base checkpoint for comparison retraining commands.",
    )
    parser.add_argument("--target-pages", type=int, default=1000)
    parser.add_argument("--negative-ratio", type=float, default=0.15)
    parser.add_argument("--pages-per-book", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pdf-dpi", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--prepare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Actually run omr_prepare_review_book.py for each planned batch.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace the campaign directory if it already exists.",
    )
    return parser.parse_args()


def page_count(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in omr_io.IMAGE_EXTENSIONS:
        return 1
    if suffix in omr_io.PDF_EXTENSIONS:
        pdfium = omr_io.require_pdfium()
        document = pdfium.PdfDocument(str(path))
        return len(document)
    raise ValueError(f"Unsupported input file: {path}")


def collect_books(inputs: list[Path], category: str) -> list[BookInfo]:
    books: list[BookInfo] = []
    for file_path in omr_io.iter_input_files(inputs):
        count = page_count(file_path)
        if count > 0:
            books.append(BookInfo(file_path, category, count))
    return books


def safe_name(path: Path) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in path.stem)


def allocate_windows(
    books: list[BookInfo],
    target_pages: int,
    pages_per_book: int,
    rng: random.Random,
    out_dir: Path,
    start_index: int,
) -> list[ReviewBatch]:
    if target_pages <= 0 or not books:
        return []
    if pages_per_book <= 0:
        raise SystemExit("--pages-per-book must be positive")

    shuffled = list(books)
    rng.shuffle(shuffled)
    batches: list[ReviewBatch] = []
    remaining = target_pages
    index = start_index
    cursor = 0
    while remaining > 0 and shuffled:
        book = shuffled[cursor % len(shuffled)]
        window_size = min(pages_per_book, book.page_count, remaining)
        if book.page_count <= window_size:
            start_page = 1
        else:
            start_page = rng.randint(1, book.page_count - window_size + 1)
        review_dir = (
            out_dir
            / "review"
            / f"batch-{index:04d}-{book.category}-{safe_name(book.path)}-p{start_page:04d}"
        )
        batches.append(
            ReviewBatch(
                index=index,
                category=book.category,
                source=book.path,
                start_page=start_page,
                max_pages=window_size,
                review_dir=review_dir,
            )
        )
        remaining -= window_size
        index += 1
        cursor += 1
    return batches


def prepare_command(batch: ReviewBatch, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("omr_prepare_review_book.py")),
        str(batch.source),
        "--weights",
        str(args.weights),
        "--out",
        str(batch.review_dir),
        "--start-page",
        str(batch.start_page),
        "--max-pages",
        str(batch.max_pages),
        "--pdf-dpi",
        str(args.pdf_dpi),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--imgsz",
        str(args.imgsz),
        "--device",
        str(args.device),
        "--force",
    ]


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def training_plan(args: argparse.Namespace, batches: list[ReviewBatch]) -> dict[str, Any]:
    dataset = args.out / "datasets" / f"{args.name}-reviewed-systems" / "data.yaml"
    run_root = args.out / "runs"
    current = {
        "name": f"{args.name}-from-current-checkpoint",
        "intent": "fresh fine-tune initialized from the current specialist checkpoint",
        "command": [
            sys.executable,
            str(Path(__file__).with_name("omr_train_staff_system_yolo.py")),
            "--data",
            str(dataset),
            "--model",
            str(args.weights),
            "--project",
            str(run_root),
            "--name",
            f"{args.name}-from-current-checkpoint",
            "--epochs",
            "80",
            "--imgsz",
            str(args.imgsz),
            "--device",
            "0",
            "--batch",
            "8",
        ],
    }
    plan: dict[str, Any] = {
        "export_command": [
            sys.executable,
            str(Path(__file__).with_name("omr_export_reviewed_yolo.py")),
            *[str(batch.review_dir) for batch in batches],
            "--out",
            str(dataset.parent),
            "--force",
        ],
        "recommended": current,
        "comparisons": [],
    }
    if args.base_model is not None:
        plan["comparisons"].append(
            {
                "name": f"{args.name}-from-base-checkpoint",
                "intent": (
                    "fresh fine-tune initialized from the upstream/base checkpoint, "
                    "not resumed from a previous specialist run"
                ),
                "command": [
                    sys.executable,
                    str(Path(__file__).with_name("omr_train_staff_system_yolo.py")),
                    "--data",
                    str(dataset),
                    "--model",
                    str(args.base_model),
                    "--project",
                    str(run_root),
                    "--name",
                    f"{args.name}-from-base-checkpoint",
                    "--epochs",
                    "80",
                    "--imgsz",
                    str(args.imgsz),
                    "--device",
                    "0",
                    "--batch",
                    "8",
                ],
            }
        )
    return plan


def batch_to_json(batch: ReviewBatch, command: list[str]) -> dict[str, Any]:
    return {
        "index": batch.index,
        "category": batch.category,
        "source": str(batch.source),
        "start_page": batch.start_page,
        "max_pages": batch.max_pages,
        "review_dir": str(batch.review_dir),
        "prepare_command": command,
        "prepare_command_text": shell_join(command),
    }


def main() -> int:
    args = parse_args()
    if args.target_pages <= 0:
        raise SystemExit("--target-pages must be positive")
    if not 0.0 <= args.negative_ratio < 1.0:
        raise SystemExit("--negative-ratio must be between 0 and 1")
    if not args.weights.exists():
        raise SystemExit(f"Missing weights: {args.weights}")
    if args.base_model is not None and not args.base_model.exists():
        raise SystemExit(f"Missing base model: {args.base_model}")
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"Output already exists: {args.out}. Pass --force to replace it.")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    score_books = collect_books(args.score_inputs, "score")
    negative_books = collect_books(args.negative_input, "negative")
    if not score_books:
        raise SystemExit("No supported score inputs found.")

    negative_target = round(args.target_pages * args.negative_ratio)
    if not negative_books:
        negative_target = 0
    score_target = args.target_pages - negative_target
    rng = random.Random(args.seed)
    score_batches = allocate_windows(
        score_books,
        score_target,
        args.pages_per_book,
        rng,
        args.out,
        1,
    )
    negative_batches = allocate_windows(
        negative_books,
        negative_target,
        args.pages_per_book,
        rng,
        args.out,
        len(score_batches) + 1,
    )
    batches = score_batches + negative_batches
    batch_json = [batch_to_json(batch, prepare_command(batch, args)) for batch in batches]
    manifest = {
        "version": CAMPAIGN_VERSION,
        "id": str(uuid.uuid4()),
        "name": args.name,
        "seed": args.seed,
        "target_pages": args.target_pages,
        "planned_pages": sum(batch.page_count for batch in batches),
        "negative_ratio": args.negative_ratio,
        "pages_per_book": args.pages_per_book,
        "model": {
            "weights": str(args.weights),
            "base_model": str(args.base_model) if args.base_model else None,
            "confidence": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
        },
        "sources": {
            "score_books": [
                {"path": str(book.path), "pages": book.page_count} for book in score_books
            ],
            "negative_books": [
                {"path": str(book.path), "pages": book.page_count} for book in negative_books
            ],
        },
        "batches": batch_json,
        "training": training_plan(args, batches),
    }
    (args.out / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.prepare:
        for batch in batch_json:
            print(batch["prepare_command_text"], file=sys.stderr)
            subprocess.run(batch["prepare_command"], check=True)

    print(
        json.dumps(
            {
                "campaign": str(args.out / "campaign.json"),
                "batches": len(batches),
                "planned_pages": manifest["planned_pages"],
                "score_pages": sum(batch.page_count for batch in score_batches),
                "negative_pages": sum(batch.page_count for batch in negative_batches),
                "prepared": args.prepare,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
