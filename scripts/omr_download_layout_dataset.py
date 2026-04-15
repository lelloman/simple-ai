#!/usr/bin/env python3
"""Download the OMR Layout Analysis YOLO dataset archives.

The upstream dataset release contains YOLO-format archives with five classes:

    0 system_measure
    1 staff_measure
    2 staff
    3 system
    4 grand_staff

For SimpleAI's current experiment we later filter this down to staff + system.
This downloader keeps the raw upstream archives intact under --raw-dir.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


DATASET_RELEASE_API = (
    "https://api.github.com/repos/v-dvorak/omr-layout-analysis/releases/tags/datasets-release"
)
DEFAULT_ROOT = Path("/tmp/simple-ai-omr-training")
NON_COCO_ARCHIVES = {
    "AudioLabs_v2_GS.tar.gz",
    "Muscima++_GS.tar.gz",
    "OSLiC.tar.gz",
    "OSLiC2.tar.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OLA layout dataset archives.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract archives after download. Default: true",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional archive names to download, e.g. Muscima++_GS.tar.gz",
    )
    return parser.parse_args()


def release_assets() -> dict[str, str]:
    with urllib.request.urlopen(DATASET_RELEASE_API, timeout=30) as response:
        release = json.load(response)
    assets = {}
    for asset in release.get("assets", []):
        name = asset.get("name")
        url = asset.get("browser_download_url")
        if name and url:
            assets[name] = url
    return assets


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"Already downloaded: {output}")
        return

    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [curl, "-L", "--fail", "--continue-at", "-", "--output", str(output), url],
            check=True,
        )
        return

    print(f"Downloading {url} -> {output}", file=sys.stderr)
    urllib.request.urlretrieve(url, output)


def safe_extract(archive: Path, output_dir: Path) -> None:
    marker = output_dir / ".extracted"
    if marker.exists():
        print(f"Already extracted: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {output_dir}", file=sys.stderr)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = output_dir / member.name
            if not target.resolve().is_relative_to(output_dir.resolve()):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        tar.extractall(output_dir)
    marker.write_text("ok\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_dir = args.root / "raw"
    extracted_dir = args.root / "extracted"
    assets = release_assets()

    wanted = set(args.only) if args.only else NON_COCO_ARCHIVES
    missing = wanted.difference(assets)
    if missing:
        raise SystemExit(f"Dataset release does not contain: {sorted(missing)}")

    for name in sorted(wanted):
        archive = raw_dir / name
        download(assets[name], archive)
        if args.extract:
            safe_name = name.removesuffix(".tar.gz").replace("+", "plus")
            safe_extract(archive, extracted_dir / safe_name)

    print(f"Dataset root: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
