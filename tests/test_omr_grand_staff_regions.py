#!/usr/bin/env python3
"""Tests for experimental grand-staff ownership region geometry."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "omr_predict_staff_system_yolo",
    SCRIPTS / "omr_predict_staff_system_yolo.py",
)
assert spec is not None
omr_predict = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = omr_predict
spec.loader.exec_module(omr_predict)


def detection(class_name: str, bbox: list[float], confidence: float = 0.9) -> dict:
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,
    }


class GrandStaffRegionTests(unittest.TestCase):
    def test_uses_midpoints_between_adjacent_boxes(self) -> None:
        regions = omr_predict.compute_grand_staff_ownership_regions(
            [
                detection("grand_staff", [100, 100, 900, 200], 0.8),
                detection("grand_staff", [100, 400, 900, 500], 0.9),
                detection("grand_staff", [100, 700, 900, 800], 0.7),
            ],
            page_width=1000,
            page_height=1000,
            x_margin_ratio=0,
            y_margin_ratio=0,
        )

        self.assertEqual([region.index for region in regions], [1, 2, 3])
        self.assertEqual(regions[0].region_bbox, [100, 100, 900, 300])
        self.assertEqual(regions[1].region_bbox, [100, 300, 900, 600])
        self.assertEqual(regions[2].region_bbox, [100, 600, 900, 800])

    def test_sorts_by_vertical_center(self) -> None:
        regions = omr_predict.compute_grand_staff_ownership_regions(
            [
                detection("grand_staff", [100, 700, 900, 800], 0.7),
                detection("grand_staff", [100, 100, 900, 200], 0.8),
                detection("grand_staff", [100, 400, 900, 500], 0.9),
            ],
            page_width=1000,
            page_height=1000,
            x_margin_ratio=0,
            y_margin_ratio=0,
        )

        self.assertEqual([region.confidence for region in regions], [0.8, 0.9, 0.7])

    def test_expands_to_page_music_area(self) -> None:
        regions = omr_predict.compute_grand_staff_ownership_regions(
            [
                detection("system_measure", [50, 80, 950, 220]),
                detection("system_measure", [40, 380, 960, 520]),
                detection("grand_staff", [100, 100, 900, 200]),
                detection("grand_staff", [120, 400, 880, 500]),
            ],
            page_width=1000,
            page_height=1000,
            x_margin_ratio=0.01,
            y_margin_ratio=0.02,
        )

        self.assertEqual(regions[0].region_bbox, [30, 60, 970, 300])
        self.assertEqual(regions[1].region_bbox, [30, 300, 970, 540])

    def test_empty_without_grand_staff_detections(self) -> None:
        regions = omr_predict.compute_grand_staff_ownership_regions(
            [detection("system", [100, 100, 900, 200])],
            page_width=1000,
            page_height=1000,
        )

        self.assertEqual(regions, [])


if __name__ == "__main__":
    unittest.main()
