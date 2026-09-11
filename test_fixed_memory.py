#!/usr/bin/env python3
"""
Fixed TRG Memory System - Refactored with modular architecture

This script now uses the following modules:
- memory.memory_builder: Memory construction and indexing
- memory.query_engine: Query execution and retrieval
- memory.test_harness: Testing and evaluation

The core logic has been moved to proper modules under memory/ folder.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter, defaultdict
import warnings
import hashlib

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_dataset import load_locomo_dataset
from memory.memory_builder import MemoryBuilder
from memory.query_engine import QueryEngine
from memory.test_harness import TestHarness
from memory.evaluator import Evaluator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format wall-clock seconds as HH:MM:SS."""
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def select_balanced_questions(questions, categories, limit):
    """Select up to limit questions with near-equal deterministic category quotas."""
    if not limit or not categories:
        return list(questions)

    ordered_categories = sorted(categories)
    base, remainder = divmod(limit, len(ordered_categories))
    quotas = {
        category: base + (index < remainder)
        for index, category in enumerate(ordered_categories)
    }
    selected = []
    selected_ids = set()
    counts = Counter()
    for question in questions:
        category = int(question.category)
        if category in quotas and counts[category] < quotas[category]:
            selected.append(question)
            selected_ids.add(id(question))
            counts[category] += 1

    # If a category has fewer available questions than its quota, fill the
    # remaining slots in original dataset order from the other categories.
    if len(selected) < limit:
        for question in questions:
            if id(question) not in selected_ids:
                selected.append(question)
                selected_ids.add(id(question))
                if len(selected) == limit:
                    break

    # Restore source order so all relation modes receive the identical sequence.
    return [question for question in questions if id(question) in selected_ids]

def score_only_mode(args):
    """
    Re-score existing results without rebuilding memory or querying.

    This mode loads existing result files and re-evaluates them with
    potentially different scoring models or methods.
    """
    import os
    from utils.memory_layer import LLMController
    from tqdm import tqdm

    print("\n" + "="*70)
    print("  SCORE-ONLY MODE: Re-evaluating existing results")
    print("="*70)

    api_key = os.getenv('OPENAI_API_KEY')
    if args.llm_backend == 'openai' and not api_key:
        print("Error: OPENAI_API_KEY not found. Cannot perform LLM-based evaluation.")
        return 1

    llm_controller = LLMController(
        backend=args.llm_backend,
        model=args.model,
        api_key=api_key if args.llm_backend == 'openai' else None,
        base_url=args.llm_base_url if args.llm_backend == 'local' else None
    )
    if args.llm_backend == 'local':
        try:
            available_models = [item.id for item in llm_controller.llm.client.models.list().data]
        except Exception as exc:
            print(f"Error: local LLM endpoint is unavailable at {args.llm_base_url}: {exc}")
            return 1
        if args.model not in available_models:
            print(f"Error: local model {args.model!r} is not served; available: {available_models}")
            return 1

    evaluator = Evaluator(llm_controller=llm_controller, use_llm_judge=True)
    print(f"Evaluator initialized with model: {args.model}\n")

    all_sample_results = []

    for sample_id in args.sample:
        print(f"\n{'='*70}")
        print(f"Re-scoring Sample {sample_id}")
        print(f"{'='*70}")

        if args.input_results:
            input_file = args.input_results
        else:
            embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
            input_file = str(
                Path(args.results_dir)
                / f"{args.relation_mode}_sample{sample_id}{embedding_suffix}.json"
            )

        if not Path(input_file).exists():
            print(f"Error: Results file not found: {input_file}")
            print(f"Please run the full test first or specify --input-results")
            continue

        with open(input_file, 'r') as f:
            data = json.load(f)

        existing_results = data.get('results', [])
        print(f"Loaded {len(existing_results)} results from {input_file}")

        updated_results = []
        print("Re-evaluating answers...")

        for result in tqdm(existing_results, desc="Re-scoring"):
            question = result['question']
            expected = result['expected']
            predicted = result['predicted']
            category = result.get('category')

            eval_result = evaluator.evaluate_answer(
                question,
                str(expected) if expected else "",
                predicted,
                question_category=category
            )

            result['metrics'] = eval_result.get('metrics')
            result['correct'] = eval_result.get('is_correct', False)
            result['llm_judge_score'] = eval_result.get('llm_judge_score', 0.0)

            updated_results.append(result)

        total = len(updated_results)
        correct = sum(1 for r in updated_results if r['correct'])
        not_found = sum(1 for r in updated_results if r['predicted'] == "Information not found")

        f1_scores = [r['metrics']['f1'] for r in updated_results if r.get('metrics') and 'f1' in r['metrics']]
        avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0

        bleu1_scores = [r['metrics']['bleu1'] for r in updated_results if r.get('metrics') and 'bleu1' in r['metrics']]
        avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) * 100 if bleu1_scores else 0

        llm_scores = [r['llm_judge_score'] for r in updated_results if r.get('llm_judge_score', 0) >= 0]
        avg_llm_score = sum(llm_scores) / len(llm_scores) * 100 if llm_scores else 0

        results_no_cat5 = [r for r in updated_results if r.get('category') != 5]
        if results_no_cat5:
            total_no_cat5 = len(results_no_cat5)
            correct_no_cat5 = sum(1 for r in results_no_cat5 if r['correct'])
            f1_no_cat5 = sum(r['metrics']['f1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            bleu1_no_cat5 = sum(r['metrics']['bleu1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            llm_no_cat5 = sum(r['llm_judge_score'] for r in results_no_cat5 if r.get('llm_judge_score', 0) >= 0) / total_no_cat5 * 100
        else:
            total_no_cat5 = correct_no_cat5 = f1_no_cat5 = bleu1_no_cat5 = llm_no_cat5 = 0

        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'f1': [], 'bleu1': [], 'llm': []})
        for r in updated_results:
            cat = r.get('category', 0)
            category_stats[cat]['total'] += 1
            if r['correct']:
                category_stats[cat]['correct'] += 1
            if r.get('metrics'):
                category_stats[cat]['f1'].append(r['metrics'].get('f1', 0))
                category_stats[cat]['bleu1'].append(r['metrics'].get('bleu1', 0))
            if r.get('llm_judge_score', 0) >= 0:
                category_stats[cat]['llm'].append(r['llm_judge_score'])

        print(f"\n{'='*70}")
        print(f"Sample {sample_id} Re-scored Results (All Categories):")
        print(f"  Total Questions: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {correct/total*100:.1f}%")
        print(f"  Average F1: {avg_f1:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1:.1f}%")
        print(f"  Average LLM Judge Score: {avg_llm_score:.1f}%")
        print(f"  Information not found: {not_found} ({not_found/total*100:.1f}%)")

        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Re-scored Results WITHOUT Category 5:")
        if results_no_cat5:
            print(f"  Total Questions: {total_no_cat5}")
            print(f"  Correct: {correct_no_cat5}")
            print(f"  Accuracy: {correct_no_cat5/total_no_cat5*100:.1f}%")
            print(f"  Average F1: {f1_no_cat5:.1f}%")
            print(f"  Average BLEU-1: {bleu1_no_cat5:.1f}%")
            print(f"  Average LLM Judge Score: {llm_no_cat5:.1f}%")

        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Re-scored Results BY CATEGORY:")
        print(f"  {'Cat':<5} {'Total':<7} {'Correct':<8} {'Acc%':<7} {'F1%':<7} {'BLEU%':<7} {'LLM%':<7}")
        print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            print(f"  {cat:<5} {stats['total']:<7} {stats['correct']:<8} {acc:<7.1f} {avg_f1_cat:<7.1f} {avg_bleu1_cat:<7.1f} {avg_llm_cat:<7.1f}")

        model_name_normalized = args.model.replace(".", "_").replace("-", "_")
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)
        output_file = f"{results_dir}/{args.relation_mode}_sample{sample_id}_rescored.json"
        category_breakdown = {}
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            category_breakdown[f"category_{cat}"] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': acc,
                'avg_f1': avg_f1_cat,
                'avg_bleu1': avg_bleu1_cat,
                'avg_llm': avg_llm_cat
            }

        with open(output_file, 'w') as f:
            json.dump({
                'sample_id': sample_id,
                'timestamp': datetime.now().isoformat(),
                'rescoring_model': args.model,
                'original_file': input_file,
                'results': updated_results,
                'stats': {
                    'overall': {
                        'total': total,
                        'correct': correct,
                        'accuracy': correct/total*100,
                        'avg_f1': avg_f1,
                        'avg_bleu1': avg_bleu1,
                        'avg_llm': avg_llm_score,
                        'not_found': not_found
                    },
                    'without_category5': {
                        'total': total_no_cat5,
                        'correct': correct_no_cat5,
                        'accuracy': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
                        'avg_f1': f1_no_cat5,
                        'avg_bleu1': bleu1_no_cat5,
                        'avg_llm': llm_no_cat5
                    },
                    'category_breakdown': category_breakdown
                }
            }, f, indent=2, default=str)

        print(f"\nRe-scored results saved to {output_file}")

        wrong_answers = [r for r in updated_results if r.get('llm_judge_score', 0) < 0.5]
        if wrong_answers:
            wrong_output_file = f"{results_dir}/{args.relation_mode}_sample{sample_id}_rescored_wrong.json"

            category_names = {
                1: "Multi-hop",
                2: "Temporal",
                3: "Open-domain",
                4: "Single-hop",
                5: "Adversarial"
            }

            formatted_wrong = []
            for wa in wrong_answers:
                formatted_wrong.append({
                    'q_id': wa['question_id'],
                    'category': f"{wa['category']} ({category_names.get(wa['category'], 'Unknown')})",
                    'question': wa['question'],
                    'expected': wa['expected'],
                    'predicted': wa['predicted'],
                    'f1': wa.get('metrics', {}).get('f1', 0),
                    'llm_score': wa.get('llm_judge_score', 0),
                    'top_nodes': wa.get('search_details', {}).get('top_nodes', []),
                    'answer_context': wa.get('answer_context', '')
                })

            wrong_summary = {
                'sample_id': sample_id,
                'total_questions': len(updated_results),
                'total_wrong': len(wrong_answers),
                'wrong_percentage': len(wrong_answers) / len(updated_results) * 100 if updated_results else 0,
                'wrong_questions': formatted_wrong
            }

            with open(wrong_output_file, 'w') as f:
                json.dump(wrong_summary, f, indent=2, default=str)

            print(f"Wrong answers ({len(wrong_answers)}) saved to {wrong_output_file}")

        all_sample_results.append({
            'sample_id': sample_id,
            'total': total,
            'correct': correct,
            'accuracy': correct/total*100,
            'avg_f1': avg_f1,
            'avg_bleu1': avg_bleu1,
            'avg_llm': avg_llm_score,
            'accuracy_no_cat5': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
            'category_breakdown': category_breakdown
        })

    if len(args.sample) > 1:
        print(f"\n\n{'='*70}")
        print(f"AGGREGATE RE-SCORED RESULTS ACROSS {len(args.sample)} SAMPLES")
        print(f"{'='*70}\n")

        avg_accuracy = sum(r['accuracy'] for r in all_sample_results) / len(all_sample_results)
        avg_f1_overall = sum(r['avg_f1'] for r in all_sample_results) / len(all_sample_results)
        avg_bleu1_overall = sum(r['avg_bleu1'] for r in all_sample_results) / len(all_sample_results)
        avg_llm_overall = sum(r['avg_llm'] for r in all_sample_results) / len(all_sample_results)
        avg_accuracy_no_cat5 = sum(r['accuracy_no_cat5'] for r in all_sample_results) / len(all_sample_results)

        print("Per-Sample Breakdown:")
        for r in all_sample_results:
            print(f"  Sample {r['sample_id']}: Acc={r['accuracy']:.1f}%, F1={r['avg_f1']:.1f}%, BLEU-1={r['avg_bleu1']:.1f}%, LLM={r['avg_llm']:.1f}%")

        print(f"\nAggregated Metrics (Average across all samples):")
        print(f"  Average Accuracy: {avg_accuracy:.1f}%")
        print(f"  Average F1: {avg_f1_overall:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1_overall:.1f}%")
        print(f"  Average LLM Judge: {avg_llm_overall:.1f}%")
        print(f"  Average Accuracy (no Cat5): {avg_accuracy_no_cat5:.1f}%")

        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)
        aggregate_output = f"{results_dir}/rescored_results_aggregate_samples_{'_'.join(map(str, args.sample))}.json"
        with open(aggregate_output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'rescoring_model': args.model,
                'samples': args.sample,
                'per_sample_results': all_sample_results,
                'aggregate': {
                    'avg_accuracy': avg_accuracy,
                    'avg_f1': avg_f1_overall,
                    'avg_bleu1': avg_bleu1_overall,
                    'avg_llm': avg_llm_overall,
                    'avg_accuracy_no_cat5': avg_accuracy_no_cat5
                }
            }, f, indent=2)
        print(f"\nAggregate re-scored results saved to {aggregate_output}")

    return 0

def main():
    """Main test function - supports multiple samples"""
    run_started = time.perf_counter()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--sample", type=int, nargs='+', default=[0],
                       help="Sample IDs to test (can specify multiple, e.g., --sample 0 1 2)")
    parser.add_argument("--max-questions", type=int, default=30,
                       help="Maximum questions per sample (default: 30)")
    parser.add_argument("--balanced-categories", action="store_true",
                       help="Select near-equal counts from requested categories")
    parser.add_argument("--cache-dir", default="./locomo_relation_experiment")
    parser.add_argument(
        "--results-dir",
        default="results_relation",
        help="Directory for per-mode and aggregate result files",
    )
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild memory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-AWQ",
                       help="LLM model name (default: local Qwen3-8B-AWQ)")
    parser.add_argument("--llm-backend", choices=["local", "openai", "ollama"],
                       default="local", help="LLM backend; local never uses OPENAI_API_KEY")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1",
                       help="OpenAI-compatible endpoint used by --llm-backend local")
    parser.add_argument("--embedding-model", type=str, default="minilm",
                       choices=["minilm", "openai"],
                       help="Embedding model to use: 'minilm' (all-MiniLM-L6-v2, 384-dim) or 'openai' (text-embedding-3-small, 1536-dim)")
    parser.add_argument("--use-episodes", action="store_true",
                       help="Use episode-based segmentation instead of turn-based (groups related turns)")
    parser.add_argument("--relation-mode", choices=["original", "free", "hybrid"],
                       default="original", help="Relation schema experiment mode")
    parser.add_argument("--score-only", action="store_true",
                       help="Only re-score existing results without rebuilding memory or querying")
    parser.add_argument("--input-results", type=str, default=None,
                       help="Input results file to re-score (used with --score-only)")
    parser.add_argument("--skip-category-5", action="store_true",
                       help="Skip category 5 (Adversarial) questions during testing")
    parser.add_argument("--category-to-test", type=str, default="1,2,3,4",
                       help="Comma-separated list of categories to test (e.g., '1,2,3,4' or '1,3'). Default: '1,2,3,4'")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel testing (parallel is enabled by default for 3x speedup)")
    parser.add_argument("--n-workers", type=int, default=3,
                       help="Number of parallel workers (default: 3)")
    parser.add_argument("--best-of-n", type=int, default=1,
                       help="Run each question N times and select best answer (default: 1)")
    parser.add_argument("--best-of-n-method", type=str, default="llm_judge",
                       choices=["llm_judge", "voting", "f1"],
                       help="Method for selecting best answer: 'llm_judge', 'voting', or 'f1' (default: llm_judge)")
    parser.add_argument("--ablation", type=str, default=None,
                       choices=["basic_retrieval", "no_causal", "no_temporal", "flat_graph"],
                       help="Run ablation study with specific configuration")
    args = parser.parse_args()

    # Defense in depth: local runs must not allow any transitive component to
    # discover or use a cloud credential inherited by the parent process.
    if args.llm_backend == 'local':
        os.environ.pop('OPENAI_API_KEY', None)

    # Parallel is enabled by default
    args.parallel = not args.no_parallel

    print("="*70)
    print("  Fixed TRG Memory System Test")
    print(f"  Model: {args.model}")
    print(f"  LLM backend: {args.llm_backend}")
    if args.llm_backend == 'local':
        print(f"  Local endpoint: {args.llm_base_url}")
    if args.score_only:
        print(f"  Mode: SCORE-ONLY (Re-evaluation)")
    else:
        print(f"  Mode: {'Episode-based' if args.use_episodes else 'Turn-based'}")
    print(f"  Samples: {args.sample}")
    print(f"  Relation mode: {args.relation_mode}")
    print(f"  Questions per sample: {args.max_questions}")
    print(f"  Balanced category sampling: {args.balanced_categories}")

    # Parse categories to test
    categories_to_test = [int(c.strip()) for c in args.category_to_test.split(',')]

    # Handle legacy skip_category_5 flag
    if args.skip_category_5 and 5 in categories_to_test:
        categories_to_test.remove(5)

    print(f"  Categories to test: {sorted(categories_to_test)}")
    category_names = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}
    print(
    "  Category types: "
    + ", ".join(
        f"{c}:{category_names.get(c, 'Unknown')}"
        for c in sorted(categories_to_test)
    )
)
    # Show parallel mode status
    # if args.parallel:
    #     print(f"  Parallel mode: ✓ ENABLED ({args.n_workers} workers, ~{args.n_workers}x speedup)")
    # else:
    #     print(f"  Parallel mode: ✗ DISABLED (use default for 3x speedup)")

    print("="*70)

    # Handle score-only mode
    if args.score_only:
        return score_only_mode(args)

    if args.llm_backend == 'local':
        if args.embedding_model == 'openai':
            print("Error: --llm-backend local requires --embedding-model minilm to avoid cloud API use.")
            return 1
        try:
            from openai import OpenAI
            local_client = OpenAI(api_key="local-vllm", base_url=args.llm_base_url)
            available_models = [item.id for item in local_client.models.list().data]
        except Exception as exc:
            print(f"Error: local LLM endpoint is unavailable at {args.llm_base_url}: {exc}")
            print("Start the vLLM server before running the experiment; no cloud fallback is used.")
            return 1
        if args.model not in available_models:
            print(f"Error: model {args.model!r} is not served by {args.llm_base_url}")
            print(f"Available local models: {available_models}")
            return 1
        print(f"Local LLM ready: {args.model}")

    # Load dataset
    samples = load_locomo_dataset(args.dataset)

    # Validate sample IDs
    for sample_id in args.sample:
        if sample_id < 0 or sample_id >= len(samples):
            print(f"Error: Sample ID {sample_id} is out of range (0-{len(samples)-1})")
            return 1

    # Process each sample
    all_sample_results = []

    for sample_idx, sample_id in enumerate(args.sample, 1):
        sample_started = time.perf_counter()
        sample = samples[sample_id]

        print(f"\n{'='*70}")
        print(f"Processing Sample {sample_id} ({sample_idx}/{len(args.sample)})")
        print(f"{'='*70}")

        # Auto-generate cache directory with sample number, embedding model, and LLM model
        embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
        # Normalize model name for folder (replace dots and hyphens with underscores)
        model_name_normalized = args.model.replace(".", "_").replace("-", "_")

        cache_dir = f"{args.cache_dir}/{args.relation_mode}/sample{sample_id}{embedding_suffix}"
        common_cache_dir = f"{args.cache_dir}/common/sample{sample_id}{embedding_suffix}"
        print(f"Mode cache directory: {cache_dir}")
        print(f"Shared baseline cache: {common_cache_dir}")
        print(f"Embedding model: {args.embedding_model}")

        # Initialize memory builder
        builder = MemoryBuilder(
            cache_dir=cache_dir,
            llm_model=args.model,
            use_episodes=args.use_episodes,
            embedding_model=args.embedding_model,
            relation_mode=args.relation_mode,
            llm_backend=args.llm_backend,
            llm_base_url=args.llm_base_url
        )

        # Build/load a mode cache. On rebuild, Free and Hybrid start from the
        # same cached baseline graph so node IDs, episodes, and edge pairs match.
        cache_file = Path(cache_dir) / "graph.json"
        mode_manifest_path = Path(cache_dir) / "manifest.json"
        common_graph = Path(common_cache_dir) / "graph.json"
        manifest_path = Path(common_cache_dir) / "manifest.json"
        dataset_digest = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
        expected_manifest = {
            'dataset_sha256': dataset_digest,
            'sample_id': sample_id,
            'model': args.model,
            'llm_backend': args.llm_backend,
            'llm_base_url': args.llm_base_url if args.llm_backend == 'local' else None,
            'embedding_model': args.embedding_model,
            'use_episodes': args.use_episodes,
        }
        expected_mode_manifest = {**expected_manifest, 'relation_mode': args.relation_mode}
        mode_cache_valid = False
        if cache_file.exists() and mode_manifest_path.exists():
            try:
                mode_cache_valid = json.loads(mode_manifest_path.read_text()) == expected_mode_manifest
            except (OSError, json.JSONDecodeError):
                mode_cache_valid = False

        if mode_cache_valid and not args.rebuild:
            logger.info("Loading cached memory...")
            builder.load()
        else:
            common_valid = False
            if common_graph.exists() and manifest_path.exists():
                try:
                    common_valid = json.loads(manifest_path.read_text()) == expected_manifest
                except (OSError, json.JSONDecodeError):
                    common_valid = False

            # Rebuilding Original refreshes the shared baseline. Other modes
            # consume it when compatible, or create it if this is the first run.
            if common_valid and args.relation_mode != 'original':
                builder.cache_dir = Path(common_cache_dir)
                builder.load()
            else:
                logger.info("Building shared baseline memory...")
                builder.build_memory(sample)
                builder.cache_dir = Path(common_cache_dir)
                builder.save()
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(expected_manifest, indent=2))

            builder.cache_dir = Path(cache_dir)
            enriched_edges = builder.apply_relation_strategy()
            print(f"Relation-enriched edges: {enriched_edges}")
            builder.save()
            mode_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            mode_manifest_path.write_text(json.dumps(expected_mode_manifest, indent=2))

        # Get memory stats
        mem_stats = builder.trg.get_statistics()
        print(f"\nMemory Statistics:")
        print(f"  Total nodes: {mem_stats['total_nodes']}")
        total_links = mem_stats.get('total_links', mem_stats['links_created'])
        print(f"  Total links: {total_links}")
        print(f"  Total vectors: {mem_stats['total_vectors']}")
        if mem_stats['total_nodes'] > 0:
            print(f"  Links per node: {total_links/mem_stats['total_nodes']:.1f}")
        print(f"  Node types: {mem_stats['node_types']}")
        print(f"  Link types: {mem_stats['link_types']}")

        # Initialize query engine with entity-session mapping
        # Prepare ablation configuration
        ablation_config = {}
        if args.ablation:
            ablation_config[args.ablation] = True
            print(f"  ⚠️ ABLATION MODE: {args.ablation}")

        query_engine = QueryEngine(
            builder.trg,
            builder.node_index,
            entity_session_map=builder.entity_session_map if hasattr(builder, 'entity_session_map') else None,
            entity_dia_map=builder.entity_dia_map if hasattr(builder, 'entity_dia_map') else None,
            ablation_config=ablation_config
        )

        # Initialize evaluator (separate from memory and query)
        evaluator = Evaluator(
            llm_controller=builder.llm_controller,
            use_llm_judge=True
        )

        # Initialize test harness with evaluator
        tester = TestHarness(builder, query_engine, evaluator=evaluator)

        # Setup best-of-N if requested
        if args.best_of_n > 1:
            from memory.best_of_n_selector import BestOfNSelector
            # print(f"\n✓ Using Best-of-{args.best_of_n} selection (method: {args.best_of_n_method})")
            tester.best_of_n = args.best_of_n
            tester.best_of_n_method = args.best_of_n_method
            tester.best_of_n_selector = BestOfNSelector(
                n_attempts=args.best_of_n,
                selection_method=args.best_of_n_method
            )

        # Filter questions based on categories to test
        original_count = len(sample.qa)
        sample.qa = [qa for qa in sample.qa if qa.category in categories_to_test]
        filtered_count = len(sample.qa)

        if args.balanced_categories:
            sample.qa = select_balanced_questions(
                sample.qa, categories_to_test, args.max_questions
            )
            selected_counts = Counter(int(qa.category) for qa in sample.qa)
            print(
                "Balanced question selection: "
                + ", ".join(
                    f"Category {category}={selected_counts.get(category, 0)}"
                    for category in sorted(categories_to_test)
                )
            )

        if filtered_count < original_count:
            print(f"\nFiltered questions: {original_count} → {filtered_count}")
            print(f"Testing categories: {sorted(categories_to_test)}\n")
        else:
            print(f"\nTesting all {filtered_count} questions\n")

        # Test questions (parallel is now default)
        test_started = time.perf_counter()
        if args.parallel:
            results = tester.test_questions_parallel(sample, args.max_questions, n_workers=args.n_workers)
        else:
            results = tester.test_questions(sample, args.max_questions)
        test_elapsed = time.perf_counter() - test_started

        # Calculate results for this sample
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        not_found = sum(1 for r in results if r['predicted'] == "Information not found")

        # Calculate average scores
        f1_scores = [r['metrics']['f1'] for r in results if r.get('metrics') and 'f1' in r['metrics']]
        avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0

        bleu1_scores = [r['metrics']['bleu1'] for r in results if r.get('metrics') and 'bleu1' in r['metrics']]
        avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) * 100 if bleu1_scores else 0

        llm_scores = [r['llm_judge_score'] for r in results if r.get('llm_judge_score', 0) >= 0]
        avg_llm_score = sum(llm_scores) / len(llm_scores) * 100 if llm_scores else 0

        # Calculate scores WITHOUT Category 5
        results_no_cat5 = [r for r in results if r.get('category') != 5]
        if results_no_cat5:
            total_no_cat5 = len(results_no_cat5)
            correct_no_cat5 = sum(1 for r in results_no_cat5 if r['correct'])
            f1_no_cat5 = sum(r['metrics']['f1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            bleu1_no_cat5 = sum(r['metrics']['bleu1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            llm_no_cat5 = sum(r['llm_judge_score'] for r in results_no_cat5 if r.get('llm_judge_score', 0) >= 0) / total_no_cat5 * 100
        else:
            total_no_cat5 = correct_no_cat5 = f1_no_cat5 = bleu1_no_cat5 = llm_no_cat5 = 0

        # Calculate category-wise breakdown
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'f1': [], 'bleu1': [], 'llm': []})
        for r in results:
            cat = r.get('category', 0)
            category_stats[cat]['total'] += 1
            if r['correct']:
                category_stats[cat]['correct'] += 1
            if r.get('metrics'):
                category_stats[cat]['f1'].append(r['metrics'].get('f1', 0))
                category_stats[cat]['bleu1'].append(r['metrics'].get('bleu1', 0))
            if r.get('llm_judge_score', 0) >= 0:
                category_stats[cat]['llm'].append(r['llm_judge_score'])

        # Print results for this sample
        print(f"\n{'='*70}")
        print(f"Sample {sample_id} Results (All Categories):")
        print(f"  Total Questions: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {correct/total*100:.1f}%")
        print(f"  Average F1: {avg_f1:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1:.1f}%")
        print(f"  Average LLM Judge Score: {avg_llm_score:.1f}%")
        print(f"  Information not found: {not_found} ({not_found/total*100:.1f}%)")
        print(f"  Question test time: {format_duration(test_elapsed)} ({test_elapsed:.1f} seconds)")

        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Results WITHOUT Category 5:")
        if results_no_cat5:
            print(f"  Total Questions: {total_no_cat5}")
            print(f"  Correct: {correct_no_cat5}")
            print(f"  Accuracy: {correct_no_cat5/total_no_cat5*100:.1f}%")
            print(f"  Average F1: {f1_no_cat5:.1f}%")
            print(f"  Average BLEU-1: {bleu1_no_cat5:.1f}%")
            print(f"  Average LLM Judge Score: {llm_no_cat5:.1f}%")

        # Print category breakdown
        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Results BY CATEGORY:")
        print(f"  {'Cat':<5} {'Total':<7} {'Correct':<8} {'Acc%':<7} {'F1%':<7} {'BLEU%':<7} {'LLM%':<7}")
        print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            print(f"  {cat:<5} {stats['total']:<7} {stats['correct']:<8} {acc:<7.1f} {avg_f1_cat:<7.1f} {avg_bleu1_cat:<7.1f} {avg_llm_cat:<7.1f}")

        # Save per-sample results with model-specific directory
        embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
        # Create model-specific results directory
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)
        output_file = f"{results_dir}/{args.relation_mode}_sample{sample_id}{embedding_suffix}.json"
        category_breakdown = {}
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            category_breakdown[f"category_{cat}"] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': acc,
                'avg_f1': avg_f1_cat,
                'avg_bleu1': avg_bleu1_cat,
                'avg_llm': avg_llm_cat
            }

        sample_elapsed_before_save = time.perf_counter() - sample_started
        with open(output_file, 'w') as f:
            json.dump({
                'sample_id': sample_id,
                'relation_mode': args.relation_mode,
                'timestamp': datetime.now().isoformat(),
                'embedding_model': args.embedding_model,
                'llm_model': args.model,
                'llm_backend': args.llm_backend,
                'llm_base_url': args.llm_base_url if args.llm_backend == 'local' else None,
                'max_questions': args.max_questions,
                'balanced_categories': args.balanced_categories,
                'selected_category_counts': dict(
                    sorted(Counter(int(result['category']) for result in results).items())
                ),
                'timing': {
                    'question_test_seconds': test_elapsed,
                    'sample_seconds_before_save': sample_elapsed_before_save
                },
                'llm_backend': args.llm_backend,
                'llm_base_url': args.llm_base_url if args.llm_backend == 'local' else None,
                'results': results,
                'stats': {
                    'overall': {
                        'total': total,
                        'correct': correct,
                        'accuracy': correct/total*100,
                        'avg_f1': avg_f1,
                        'avg_bleu1': avg_bleu1,
                        'avg_llm': avg_llm_score,
                        'not_found': not_found
                    },
                    'without_category5': {
                        'total': total_no_cat5,
                        'correct': correct_no_cat5,
                        'accuracy': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
                        'avg_f1': f1_no_cat5,
                        'avg_bleu1': bleu1_no_cat5,
                        'avg_llm': llm_no_cat5
                    },
                    'category_breakdown': category_breakdown,
                    'memory_stats': mem_stats,
                    'relation_stats': builder.get_relation_statistics()
                }
            }, f, indent=2, default=str)

        print(f"Results saved to {output_file}")
        sample_elapsed = time.perf_counter() - sample_started
        print(
            f"Sample {sample_id} total elapsed time: "
            f"{format_duration(sample_elapsed)} ({sample_elapsed:.1f} seconds)"
        )

        # Store for aggregation
        all_sample_results.append({
            'sample_id': sample_id,
            'total': total,
            'correct': correct,
            'accuracy': correct/total*100,
            'avg_f1': avg_f1,
            'avg_bleu1': avg_bleu1,
            'avg_llm': avg_llm_score,
            'accuracy_no_cat5': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
            'question_test_seconds': test_elapsed,
            'sample_elapsed_seconds': sample_elapsed,
            'category_breakdown': category_breakdown
        })

    # Print summary if multiple samples
    if len(args.sample) > 1:
        print(f"\n\n{'='*70}")
        print(f"AGGREGATE RESULTS ACROSS {len(args.sample)} SAMPLES")
        print(f"{'='*70}\n")

        # Calculate averages
        avg_accuracy = sum(r['accuracy'] for r in all_sample_results) / len(all_sample_results)
        avg_f1_overall = sum(r['avg_f1'] for r in all_sample_results) / len(all_sample_results)
        avg_bleu1_overall = sum(r['avg_bleu1'] for r in all_sample_results) / len(all_sample_results)
        avg_llm_overall = sum(r['avg_llm'] for r in all_sample_results) / len(all_sample_results)
        avg_accuracy_no_cat5 = sum(r['accuracy_no_cat5'] for r in all_sample_results) / len(all_sample_results)

        print("Per-Sample Breakdown:")
        for r in all_sample_results:
            print(f"  Sample {r['sample_id']}: Acc={r['accuracy']:.1f}%, F1={r['avg_f1']:.1f}%, BLEU-1={r['avg_bleu1']:.1f}%, LLM={r['avg_llm']:.1f}%")

        print(f"\nAggregated Metrics (Average across all samples):")
        print(f"  Average Accuracy: {avg_accuracy:.1f}%")
        print(f"  Average F1: {avg_f1_overall:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1_overall:.1f}%")
        print(f"  Average LLM Judge: {avg_llm_overall:.1f}%")
        print(f"  Average Accuracy (no Cat5): {avg_accuracy_no_cat5:.1f}%")

        # Aggregate category breakdown across all samples
        print(f"\n{'-'*70}")
        print(f"AGGREGATE RESULTS BY CATEGORY (Average across {len(args.sample)} samples):")

        # Collect all categories across all samples
        all_categories = set()
        for r in all_sample_results:
            all_categories.update(r['category_breakdown'].keys())

        # Calculate average stats per category
        aggregate_category_stats = {}
        for cat_key in sorted(all_categories, key=lambda x: int(x.split('_')[1])):
            cat_num = int(cat_key.split('_')[1])
            total_samples_with_cat = 0
            sum_total = sum_correct = sum_acc = sum_f1 = sum_bleu = sum_llm = 0

            for r in all_sample_results:
                if cat_key in r['category_breakdown']:
                    cat_data = r['category_breakdown'][cat_key]
                    total_samples_with_cat += 1
                    sum_total += cat_data['total']
                    sum_correct += cat_data['correct']
                    sum_acc += cat_data['accuracy']
                    sum_f1 += cat_data['avg_f1']
                    sum_bleu += cat_data['avg_bleu1']
                    sum_llm += cat_data['avg_llm']

            if total_samples_with_cat > 0:
                aggregate_category_stats[cat_num] = {
                    'avg_total': sum_total / total_samples_with_cat,
                    'avg_correct': sum_correct / total_samples_with_cat,
                    'avg_accuracy': sum_acc / total_samples_with_cat,
                    'avg_f1': sum_f1 / total_samples_with_cat,
                    'avg_bleu1': sum_bleu / total_samples_with_cat,
                    'avg_llm': sum_llm / total_samples_with_cat,
                    'samples_count': total_samples_with_cat
                }

        print(f"  {'Cat':<5} {'Avg Total':<10} {'Avg Corr':<10} {'Acc%':<7} {'F1%':<7} {'BLEU%':<7} {'LLM%':<7} {'#Samples':<9}")
        print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
        for cat in sorted(aggregate_category_stats.keys()):
            stats = aggregate_category_stats[cat]
            print(f"  {cat:<5} {stats['avg_total']:<10.1f} {stats['avg_correct']:<10.1f} "
                  f"{stats['avg_accuracy']:<7.1f} {stats['avg_f1']:<7.1f} {stats['avg_bleu1']:<7.1f} "
                  f"{stats['avg_llm']:<7.1f} {stats['samples_count']:<9}")

        # Save aggregate results in model-specific directory
        embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
        # Keep aggregate outputs beside the per-mode experiment results.
        model_name_normalized = args.model.replace(".", "_").replace("-", "_")
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)
        aggregate_output = f"{results_dir}/{args.relation_mode}_aggregate_samples_{'_'.join(map(str, args.sample))}{embedding_suffix}.json"
        with open(aggregate_output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'embedding_model': args.embedding_model,
                'llm_model': args.model,
                'samples': args.sample,
                'relation_mode': args.relation_mode,
                'per_sample_results': all_sample_results,
                'aggregate': {
                    'avg_accuracy': avg_accuracy,
                    'avg_f1': avg_f1_overall,
                    'avg_bleu1': avg_bleu1_overall,
                    'avg_llm': avg_llm_overall,
                    'avg_accuracy_no_cat5': avg_accuracy_no_cat5,
                    'category_breakdown': aggregate_category_stats
                }
            }, f, indent=2)
        print(f"\nAggregate results saved to {aggregate_output}")

    run_elapsed = time.perf_counter() - run_started
    print(f"\n{'='*70}")
    print(
        f"TOTAL RUN ELAPSED TIME: {format_duration(run_elapsed)} "
        f"({run_elapsed:.1f} seconds)"
    )
    print(f"{'='*70}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
