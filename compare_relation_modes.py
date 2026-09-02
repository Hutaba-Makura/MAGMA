#!/usr/bin/env python3
"""Compare Original, Free, and Hybrid relation experiment outputs."""

import argparse
import csv
import json
from pathlib import Path


MODES = ("original", "free", "hybrid")
CATEGORY_NAMES = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}


def load_run(results_dir: Path, mode: str, sample: int):
    candidates = [
        results_dir / f"{mode}_sample{sample}.json",
        results_dir / f"{mode}_sample{sample}_openai.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text()), path
    raise FileNotFoundError(f"No result for {mode}, sample {sample} in {results_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_relation")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = {}
    for mode in MODES:
        runs[mode], path = load_run(results_dir, mode, args.sample)
        print(f"Loaded {mode}: {path}")

    rows = []
    overall_metrics = (
        ("Accuracy", "accuracy"),
        ("Average F1", "avg_f1"),
        ("BLEU-1", "avg_bleu1"),
        ("LLM Judge", "avg_llm"),
        ("Information not found", "not_found"),
    )
    for label, key in overall_metrics:
        rows.append({
            "scope": "Overall",
            "metric": label,
            **{mode: runs[mode]["stats"]["overall"].get(key, 0) for mode in MODES},
        })

    for category, name in CATEGORY_NAMES.items():
        cat_key = f"category_{category}"
        if not any(cat_key in runs[mode]["stats"].get("category_breakdown", {}) for mode in MODES):
            continue
        for label, key in (("Accuracy", "accuracy"), ("Average F1", "avg_f1")):
            rows.append({
                "scope": name,
                "metric": label,
                **{
                    mode: runs[mode]["stats"].get("category_breakdown", {}).get(cat_key, {}).get(key, 0)
                    for mode in MODES
                },
            })

    print("\n{:<18} {:<24} {:>12} {:>12} {:>12}".format(
        "Scope", "Metric", "Original", "Free", "Hybrid"
    ))
    for row in rows:
        print("{:<18} {:<24} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            row["scope"], row["metric"],
            float(row["original"]), float(row["free"]), float(row["hybrid"]),
        ))

    relation_stats = {
        mode: runs[mode]["stats"].get("relation_stats", {}) for mode in MODES
    }
    topology_hashes = {
        mode: relation_stats[mode].get("topology_sha256") for mode in MODES
    }
    topology_identical = len(set(topology_hashes.values())) == 1 and None not in topology_hashes.values()
    print(f"\nTopology identical across modes: {topology_identical}")
    print("\nRelation statistics:")
    print(json.dumps(relation_stats, indent=2))

    prefix = Path(args.output_prefix) if args.output_prefix else results_dir / f"comparison_sample{args.sample}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(json.dumps({
        "sample_id": args.sample,
        "metrics": rows,
        "topology_identical": topology_identical,
        "topology_hashes": topology_hashes,
        "relation_stats": relation_stats,
    }, indent=2))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "metric", *MODES])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
