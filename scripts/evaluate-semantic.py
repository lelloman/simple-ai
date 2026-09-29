#!/usr/bin/env python3
"""Small labelled semantic evaluation; measures calibration without fitting it."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from simple_ai_semantic import arguments, from_args, json_safe


def metrics(rows):
    labelled = [row for row in rows if row["expected"] is not None]
    if not labelled:
        return {"labelled_count": 0}
    n = len(labelled)
    correct = [row["selected"] == row["expected"] for row in labelled]
    confidences = [max(row["scores"].values()) for row in labelled]
    bins = []
    ece = 0.0
    for index in range(10):
        members = [i for i, c in enumerate(confidences) if min(int(c * 10), 9) == index]
        if not members:
            continue
        confidence = sum(confidences[i] for i in members) / len(members)
        accuracy = sum(correct[i] for i in members) / len(members)
        ece += len(members) / n * abs(accuracy - confidence)
        bins.append({"lower": index / 10, "upper": (index + 1) / 10, "count": len(members),
                     "mean_confidence": confidence, "accuracy": accuracy})
    selective = []
    for threshold in [.5, .7, .9, .95, .99]:
        accepted = [i for i, c in enumerate(confidences) if c >= threshold]
        selective.append({"threshold": threshold, "coverage": len(accepted) / n,
                          "accuracy": sum(correct[i] for i in accepted) / len(accepted) if accepted else None})
    binary = [r for r in labelled if r["kind"] == "binary"]
    return {"labelled_count": n, "accuracy": sum(correct) / n,
            "multiclass_brier": sum(sum((v - float(k == r["expected"])) ** 2 for k, v in r["scores"].items()) for r in labelled) / n,
            "binary_brier": sum((r["scores"]["True"] - float(r["expected"] == "True")) ** 2 for r in binary) / len(binary) if binary else None,
            "nll": sum(-math.log(max(r["scores"][r["expected"]], 1e-15)) for r in labelled) / n,
            "ece_10_equal_width_bins": ece, "reliability_bins": bins, "selective_accuracy": selective}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/semantic-quality.json")
    parser.add_argument("--output", type=Path, default=Path("semantic-quality-results.json"))
    args = parser.parse_args()
    scorer = from_args(args)
    cases = json.loads(args.dataset.read_text())
    rows = []
    for case in cases:
        if case["kind"] == "binary":
            result = scorer.semantic_score(case["state"], case["predicate"])
        else:
            result = scorer.semantic_classify(case["state"], case["options"])
        rows.append({**case, **result})
    groups = sorted(set(row["category"] for row in rows))
    ambiguous = [r for r in rows if r["expected"] is None]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "model": args.model, "revision": args.revision,
              "calibrated": False, "dataset": str(args.dataset), "metrics": metrics(rows),
              "by_category": {group: metrics([r for r in rows if r["category"] == group]) for group in groups},
              "ambiguous": {"count": len(ambiguous), "high_confidence_count": sum(max(r["scores"].values()) >= .9 for r in ambiguous),
                            "note": "No determinate ground truth; excluded from accuracy/calibration. High confidence is a diagnostic only."},
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
