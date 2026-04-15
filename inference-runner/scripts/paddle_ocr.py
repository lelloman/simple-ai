#!/usr/bin/env python3
"""PaddleOCR CLI adapter for simple-ai-runner.

Contract:
  paddle_ocr.py --input <file> --filename <name> --options <json> --output <json>

The script emits the shared SimpleAI OCR JSON schema. It intentionally keeps
provider-specific details inside this adapter so the Rust runner can swap OCR
implementations without changing the HTTP API.
"""

from __future__ import annotations

import argparse
import json
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._in_cell = False
        self._cell_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._in_cell = True
            self._cell_text = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell and self._current_row is not None:
            self._current_row.append("".join(self._cell_text).strip())
            self._in_cell = False
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None


def _parse_html_table(html: str | None) -> list[list[str]]:
    if not html:
        return []
    parser = TableParser()
    parser.feed(html)
    return parser.rows


def _bbox_from_points(points: Any) -> list[float] | None:
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _load_images(input_path: Path) -> list[tuple[int, Any]]:
    suffix = input_path.suffix.lower()
    if suffix != ".pdf":
        return [(1, str(input_path))]

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("PDF OCR requires pypdfium2 to be installed") from exc

    document = pdfium.PdfDocument(str(input_path))
    pages: list[tuple[int, Any]] = []
    for index in range(len(document)):
        bitmap = document[index].render(scale=2).to_pil()
        pages.append((index + 1, bitmap))
    return pages


def _ocr_page(ocr: Any, image: Any, page_num: int) -> dict[str, Any]:
    result = ocr.ocr(image, cls=True)
    lines = result[0] if result else []
    blocks: list[dict[str, Any]] = []

    for line in lines:
        points = line[0]
        text = line[1][0]
        confidence = float(line[1][1])
        blocks.append(
            {
                "kind": "text",
                "text": text,
                "confidence": confidence,
                "bbox": _bbox_from_points(points),
            }
        )

    page_text = "\n".join(block["text"] for block in blocks)
    return {
        "page": page_num,
        "text": page_text,
        "blocks": blocks,
        "tables": [],
    }


def _structure_page(engine: Any, image: Any, page_num: int) -> dict[str, Any]:
    result = engine(image)
    blocks: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []

    for item in result or []:
        kind = str(item.get("type") or "text")
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox = [float(v) for v in bbox]
        else:
            bbox = None

        if kind == "table":
            res = item.get("res") or {}
            rows = _parse_html_table(res.get("html") if isinstance(res, dict) else None)
            tables.append(
                {
                    "bbox": bbox,
                    "rows": rows,
                    "confidence": None,
                }
            )
            continue

        res = item.get("res")
        if isinstance(res, list):
            text_parts = []
            for line in res:
                if isinstance(line, dict):
                    text_parts.append(str(line.get("text") or ""))
                elif isinstance(line, (list, tuple)) and len(line) > 1:
                    text_parts.append(str(line[1][0] if isinstance(line[1], (list, tuple)) else line[1]))
            text = "\n".join(part for part in text_parts if part)
        elif isinstance(res, dict):
            text = str(res.get("text") or "")
        else:
            text = str(res or "")

        if text:
            blocks.append(
                {
                    "kind": kind,
                    "text": text,
                    "confidence": item.get("confidence"),
                    "bbox": bbox,
                }
            )

    page_text = "\n".join(block["text"] for block in blocks)
    return {
        "page": page_num,
        "text": page_text,
        "blocks": blocks,
        "tables": tables,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--filename")
    parser.add_argument("--options")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.health:
        try:
            import paddleocr
            from paddleocr import PaddleOCR  # noqa: F401

            try:
                from paddleocr import PPStructure  # noqa: F401
                features = ["text", "layout", "tables", "pdf"]
                modes = ["text", "layout", "document"]
            except Exception:
                features = ["text", "layout", "pdf"]
                modes = ["text", "layout"]

            version = getattr(paddleocr, "__version__", None)
        except Exception as exc:
            raise RuntimeError(f"paddleocr health check failed: {exc}") from exc

        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "provider": "paddleocr",
                    "providerVersion": version,
                    "modes": modes,
                    "features": features,
                    "languages": [],
                },
                handle,
            )
        return

    if not args.input or not args.options:
        raise RuntimeError("--input and --options are required unless --health is used")

    started = time.monotonic()
    input_path = Path(args.input)
    with open(args.options, "r", encoding="utf-8") as handle:
        options = json.load(handle)

    mode = options.get("mode", "document")
    languages = options.get("languages") or ["en"]
    max_pages = options.get("maxPages")
    # PaddleOCR's Python API expects one language code for a model family.
    # The runner schema allows a list so future providers can support true
    # multilingual requests without changing the HTTP API.
    lang = languages[0]

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("paddleocr is not installed") from exc

    page_inputs = _load_images(input_path)
    if max_pages is not None:
        page_inputs = page_inputs[: int(max_pages)]

    if mode == "document":
        try:
            from paddleocr import PPStructure

            structure_engine = PPStructure(show_log=False, lang=lang)
            pages = [
                _structure_page(structure_engine, image, page_num)
                for page_num, image in page_inputs
            ]
        except Exception:
            ocr = PaddleOCR(use_angle_cls=True, lang=lang)
            pages = [_ocr_page(ocr, image, page_num) for page_num, image in page_inputs]
    else:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        pages = [_ocr_page(ocr, image, page_num) for page_num, image in page_inputs]

    blocks = [block for page in pages for block in page["blocks"]]
    tables = [table for page in pages for table in page["tables"]]
    text = "\n\n".join(page["text"] for page in pages)

    if mode == "text":
        for page in pages:
            page["blocks"] = []
            page["tables"] = []
        blocks = []
    elif mode == "layout":
        for page in pages:
            page["tables"] = []

    response = {
        "text": text,
        "pages": pages,
        "blocks": blocks,
        "tables": tables,
        "metadata": {
            "provider": "paddleocr",
            "providerVersion": None,
            "mode": mode,
            "pageCount": len(pages),
            "elapsedMs": int((time.monotonic() - started) * 1000),
        },
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(response, handle, ensure_ascii=False)


if __name__ == "__main__":
    main()
