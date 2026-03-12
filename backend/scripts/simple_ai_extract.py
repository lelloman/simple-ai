#!/usr/bin/env python3
import json
import re
import sys

import trafilatura
from trafilatura import metadata


JUNK_PATTERNS = [
    re.compile(r"\b(login|sign in|access denied|cookie|consent)\b", re.IGNORECASE),
    re.compile(r"please provide the text", re.IGNORECASE),
]


def flags_for_text(url: str, text: str) -> list[str]:
    flags: list[str] = []
    lowered = text.strip().lower()
    if len(lowered) < 200:
        flags.append("too_short")
    if any(pattern.search(lowered) for pattern in JUNK_PATTERNS):
        flags.append("junk_pattern")
    if re.search(r"/login|/utente/profilo|/account", url, re.IGNORECASE):
        flags.append("login_url")
    return flags


def main() -> int:
    request = json.load(sys.stdin)
    html = request.get("html", "")
    url = request.get("url", "")

    if not html or not url:
        json.dump(
            {
                "ok": False,
                "text": None,
                "html": None,
                "title": None,
                "author": None,
                "date": None,
                "siteName": None,
                "extractor": "trafilatura",
                "extractorVersion": trafilatura.__version__,
                "rejected": True,
                "flags": ["missing_input"],
                "score": None,
                "error": "url and html are required",
            },
            sys.stdout,
        )
        return 0

    extracted_text = trafilatura.extract(
        html,
        url=url,
        favor_precision=True,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        output_format="txt",
    )
    extracted_html = trafilatura.extract(
        html,
        url=url,
        favor_precision=True,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        output_format="html",
    )
    doc = metadata.extract_metadata(filecontent=html, default_url=url)

    text = (extracted_text or "").strip()
    html_out = (extracted_html or "").strip()
    flags = flags_for_text(url, text)
    rejected = not text or bool(flags)
    score = None if not text else min(1.0, len(text) / 4000.0)

    json.dump(
        {
            "ok": True,
            "text": text or None,
            "html": html_out or None,
            "title": getattr(doc, "title", None),
            "author": getattr(doc, "author", None),
            "date": getattr(doc, "date", None),
            "siteName": getattr(doc, "sitename", None),
            "extractor": "trafilatura",
            "extractorVersion": trafilatura.__version__,
            "rejected": rejected,
            "flags": flags,
            "score": score,
            "error": None,
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
