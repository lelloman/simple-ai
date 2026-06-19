#!/usr/bin/env python3
"""Tests for exporting reviewed OMR system boxes to YOLO format."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "omr_export_reviewed_yolo",
    SCRIPTS / "omr_export_reviewed_yolo.py",
)
assert spec is not None
omr_export = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = omr_export
spec.loader.exec_module(omr_export)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def create_review_dir(root: Path) -> Path:
    review_dir = root / "review-book"
    pages_dir = review_dir / "pages"
    reviewed_dir = review_dir / "reviewed"
    pages_dir.mkdir(parents=True)
    reviewed_dir.mkdir(parents=True)

    for index in range(1, 4):
        (pages_dir / f"page-{index:04d}.png").write_bytes(b"fake png")

    manifest_pages = []
    page_specs = [
        (1, "reviewed", "reviewed"),
        (2, "pending", "pending"),
        (3, "reviewed", "reviewed"),
    ]
    for sequence, manifest_status, review_status in page_specs:
        review_name = f"page-{sequence:04d}.json"
        manifest_pages.append(
            {
                "sequence": sequence,
                "page": sequence,
                "image": f"pages/page-{sequence:04d}.png",
                "review": f"reviewed/{review_name}",
                "status": manifest_status,
            }
        )
        boxes = []
        if sequence == 1:
            boxes = [
                {
                    "id": "accepted-system",
                    "class": "system",
                    "bbox": [100, 50, 500, 250],
                    "status": "accepted",
                },
                {
                    "id": "pending-system",
                    "class": "system",
                    "bbox": [0, 0, 100, 100],
                    "status": "pending",
                },
                {
                    "id": "accepted-staff",
                    "class": "staff",
                    "bbox": [0, 0, 100, 100],
                    "status": "accepted",
                },
            ]
        elif sequence == 2:
            boxes = [
                {
                    "id": "pending-page-system",
                    "class": "system",
                    "bbox": [100, 100, 200, 200],
                    "status": "accepted",
                }
            ]
        write_json(
            reviewed_dir / review_name,
            {
                "version": 1,
                "page": sequence,
                "sequence": sequence,
                "source": "synthetic.pdf",
                "image": f"pages/page-{sequence:04d}.png",
                "width": 1000,
                "height": 500,
                "status": review_status,
                "boxes": boxes,
                "helpers": [],
            },
        )

    write_json(
        review_dir / "manifest.json",
        {
            "version": 1,
            "source_inputs": ["synthetic.pdf"],
            "pages": manifest_pages,
        },
    )
    return review_dir


class ExportReviewedYoloTests(unittest.TestCase):
    def test_bbox_to_yolo_line_normalizes_pixels(self) -> None:
        line = omr_export.bbox_to_yolo_line([100, 50, 500, 250], 1000, 500)

        self.assertEqual(line, "0 0.30000000 0.30000000 0.40000000 0.40000000")

    def test_collect_examples_exports_reviewed_accepted_system_boxes_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            review_dir = create_review_dir(Path(tmp))
            examples = omr_export.collect_examples(
                [review_dir],
                page_statuses={"reviewed"},
                box_statuses={"accepted"},
                keep_empty=True,
            )

            self.assertEqual([example.sequence for example in examples], [1, 3])
            self.assertEqual(
                examples[0].labels,
                ["0 0.30000000 0.30000000 0.40000000 0.40000000"],
            )
            self.assertEqual(examples[1].labels, [])

    def test_export_dataset_writes_yolo_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            review_dir = create_review_dir(tmp_path)
            out_dir = tmp_path / "dataset"
            examples = omr_export.collect_examples(
                [review_dir],
                page_statuses={"reviewed"},
                box_statuses={"accepted"},
                keep_empty=True,
            )

            summary = omr_export.export_dataset(
                examples,
                out_dir,
                train_split=0.5,
                seed=1,
                copy=True,
            )

            self.assertEqual(summary["images"], 2)
            self.assertEqual(summary["boxes"], 1)
            self.assertTrue((out_dir / "data.yaml").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            label_files = sorted((out_dir / "labels").glob("*/*.txt"))
            self.assertEqual(len(label_files), 2)
            self.assertIn(
                "names:\n  0: system",
                (out_dir / "data.yaml").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "0 0.30000000 0.30000000 0.40000000 0.40000000",
                "\n".join(path.read_text(encoding="utf-8") for path in label_files),
            )


if __name__ == "__main__":
    unittest.main()
