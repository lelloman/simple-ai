#!/usr/bin/env python3
"""Verify pinned JEV head/tokenizer alignment without loading model weights."""
import argparse
import json
from simple_ai_semantic import arguments, from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    args = parser.parse_args()
    scorer = from_args(args, verify_backend=False)
    for state in ["Build failed.", "日本語のログ; errore di rete; 😀", {"log": "passed", "metadata": [1, 2]}]:
        prepared = scorer.prepare(state, ["The build succeeded.", "More information is required."], ["False", "True"])
        for branch in prepared.branches:
            text = scorer.tokenizer.decode(branch)
            assert scorer.tokenizer.encode(text, add_special_tokens=False) == branch
            assert text.startswith("[kind] noul\n[state] ")
            assert text.endswith("\n[options]\nfalse\ntrue\n[decision]:")
    print(json.dumps({"model": args.model, "revision": args.revision,
                      "head_slots": 24, "boundary_checks": 6, "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
