#!/usr/bin/env python3
"""Train a YOLO detector for OMR staff + system boxes."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DATA = Path("/tmp/simple-ai-omr-training/datasets/staff-system-yolo/data.yaml")
DEFAULT_PROJECT = Path("/tmp/simple-ai-omr-training/runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on staff+system OMR boxes.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="staff-system-yolo11n")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install with: pip install -r scripts/omr-layout-requirements.txt"
        ) from exc

    if not args.data.exists():
        raise SystemExit(f"Missing YOLO data config: {args.data}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        patience=args.patience,
        plots=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
