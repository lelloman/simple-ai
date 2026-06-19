#!/usr/bin/env python3
"""Tests for OMR review campaign planning."""

from __future__ import annotations

import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "omr_create_review_campaign",
    SCRIPTS / "omr_create_review_campaign.py",
)
assert spec is not None
omr_campaign = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = omr_campaign
spec.loader.exec_module(omr_campaign)


class CreateReviewCampaignTests(unittest.TestCase):
    def test_allocate_windows_caps_pages_per_book_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            books = [
                omr_campaign.BookInfo(Path("/scores/a.pdf"), "score", 100),
                omr_campaign.BookInfo(Path("/scores/b.pdf"), "score", 10),
            ]

            batches = omr_campaign.allocate_windows(
                books,
                target_pages=50,
                pages_per_book=24,
                rng=random.Random(1),
                out_dir=out_dir,
                start_index=1,
            )

            self.assertEqual(sum(batch.page_count for batch in batches), 50)
            self.assertTrue(all(batch.page_count <= 24 for batch in batches))
            self.assertEqual([batch.index for batch in batches], [1, 2, 3, 4])
            self.assertTrue(all("score" in str(batch.review_dir) for batch in batches))

    def test_training_plan_uses_planned_review_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            args = SimpleNamespace(
                out=out_dir,
                name="systems-v002",
                weights=Path("/models/current.pt"),
                base_model=Path("/models/base.pt"),
                imgsz=1536,
            )
            batches = [
                omr_campaign.ReviewBatch(
                    index=1,
                    category="score",
                    source=Path("/scores/a.pdf"),
                    start_page=1,
                    max_pages=12,
                    review_dir=out_dir / "review" / "batch-0001-score-a-p0001",
                )
            ]

            plan = omr_campaign.training_plan(args, batches)

            self.assertIn(str(batches[0].review_dir), plan["export_command"])
            self.assertEqual(
                plan["recommended"]["command"][plan["recommended"]["command"].index("--model") + 1],
                "/models/current.pt",
            )
            self.assertEqual(
                plan["comparisons"][0]["command"][
                    plan["comparisons"][0]["command"].index("--model") + 1
                ],
                "/models/base.pt",
            )


if __name__ == "__main__":
    unittest.main()
