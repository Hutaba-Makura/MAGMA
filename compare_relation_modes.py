#!/usr/bin/env python3
"""Compare Original, Free, and Hybrid relation experiment outputs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = ("original", "free", "hybrid")
CATEGORY_NAMES = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}
MODE_LABELS = {"original": "Fixed", "free": "Free", "hybrid": "Hybrid"}
MODE_COLORS = {"original": "#4C78A8", "free": "#F58518", "hybrid": "#54A24B"}


def load_run(results_dir: Path, mode: str, sample: int):
    candidates = [
        results_dir / f"{mode}_sample{sample}.json",
        results_dir / f"{mode}_sample{sample}_openai.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text()), path
    raise FileNotFoundError(f"No result for {mode}, sample {sample} in {results_dir}")


def save_comparison_figure(runs, output_path: Path, sample: int) -> None:
    """Save a compact publication-friendly comparison figure."""
    modes = list(MODES)
    labels = [MODE_LABELS[mode] for mode in modes]
    colors = [MODE_COLORS[mode] for mode in modes]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    metrics = [("Accuracy", "accuracy"), ("F1", "avg_f1"),
               ("BLEU-1", "avg_bleu1"), ("LLM Judge", "avg_llm")]
    x = np.arange(len(metrics))
    width = 0.24
    for index, mode in enumerate(modes):
        values = [runs[mode]["stats"]["overall"].get(key, 0) for _, key in metrics]
        axes[0].bar(x + (index - 1) * width, values, width,
                    label=labels[index], color=colors[index])
    axes[0].set_xticks(x, [label for label, _ in metrics], rotation=15)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Overall metrics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    categories = [category for category in (1, 2, 3, 4, 5)
                  if all(f"category_{category}" in runs[mode]["stats"].get("category_breakdown", {})
                         for mode in modes)]
    x = np.arange(len(categories))
    for index, mode in enumerate(modes):
        values = [runs[mode]["stats"]["category_breakdown"][f"category_{category}"]["avg_f1"]
                  for category in categories]
        axes[1].bar(x + (index - 1) * width, values, width,
                    label=labels[index], color=colors[index])
    axes[1].set_xticks(x, [CATEGORY_NAMES[category] for category in categories], rotation=18)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("F1 (%)")
    axes[1].set_title("F1 by question category")
    axes[1].grid(axis="y", alpha=0.25)

    count = len(runs[modes[0]].get("results", []))
    fig.suptitle(f"MAGMA relation schema comparison — Sample {sample}, n={count}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    category_path = output_path.with_name(f"{output_path.stem}_category_f1.png")
    category_fig, category_ax = plt.subplots(figsize=(9, 5.5))
    for index, mode in enumerate(modes):
        values = [runs[mode]["stats"]["category_breakdown"][f"category_{category}"]["avg_f1"]
                  for category in categories]
        category_ax.bar(x + (index - 1) * width, values, width,
                        label=labels[index], color=colors[index])
    category_ax.set_xticks(
        x, [CATEGORY_NAMES[category] for category in categories], rotation=12
    )
    category_ax.set_ylim(0, 100)
    category_ax.set_ylabel("F1 (%)")
    category_ax.set_title(
        f"F1 by question category — Sample {sample}, n={count}"
    )
    category_ax.grid(axis="y", alpha=0.25)
    category_ax.legend(frameon=False)
    category_fig.tight_layout()
    category_fig.savefig(category_path, dpi=200, bbox_inches="tight")
    category_fig.savefig(category_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(category_fig)


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
    figure_path = prefix.with_suffix(".png")
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
    save_comparison_figure(runs, figure_path, args.sample)
    print(f"\nSaved: {json_path}, {csv_path}, {figure_path}, and {figure_path.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
