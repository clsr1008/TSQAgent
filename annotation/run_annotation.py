"""
Pairwise annotation CLI for TSqualityAgent.

Loads a dataset's blocks.jsonl, samples random pairs, runs the agent pipeline
on each pair, and writes high-confidence results to an output JSONL file.

Target: 500 valid pairs per dataset.  The optional significance report stores
the Perceiver plan and counts how often it selects each dimension for rebuttal
analysis.
For small datasets where C(N,2) < 500, we use all possible pairs as the budget.

Confidence filter: |2*comparisons_avg - 1| >= 0.5
  (winner ≠ tie AND agent confidence >= 0.5)

Usage
-----
# Main config: finetuned perceiver + base inspector/adjudicator
python -m annotation.run_annotation \
    --dataset datasets/electricity/blocks.jsonl \
    --output datasets/electricity/annotation.jsonl \
    --dataset_description "Electricity consumption time series" \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1 \
    --n_valid 20

CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen3-4B \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768

CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen3-4B \
  --enable-lora \
  --lora-modules perceiver-grpo-v2=training/checkpoints/perceiver-grpo-v2 \
  --port 8001 \
  --max-model-len 32768

# Rebuttal audit: use a NEW output file because existing annotation.jsonl does
# not contain Perceiver/Inspector states.  This also writes a CSV beside the
# requested JSON report.
python -m annotation.run_annotation \
    --dataset datasets/electricity/blocks.jsonl \
    --output datasets/electricity/annotation_significance.jsonl \
    --dataset_description "Electricity load time series for downstream forecasting; assess data quality differences relevant to forecasting." \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1 \
    --n_valid 300 \
    --significance_report results/electricity_dimension_significance.json

# Real telemetry rebuttal: label-blind normal-vs-anomaly evaluation plus
# independent reference-based weak labels for Perceiver and Inspector.
# The default random channel selection is the conservative primary framework
# evaluation. Run a separate output with --reference_eval_channel_selection
# max_reference_evidence for the diagnostic Perceiver/Inspector study.
python -m annotation.run_annotation \
    --reference_eval_test_data datasets/MSL/MSL_test.npy \
    --reference_eval_labels datasets/MSL/MSL_test_label.npy \
    --reference_eval_reference_data datasets/MSL/MSL_train.npy \
    --reference_eval_output datasets/MSL/reference_eval_pairs.jsonl \
    --reference_eval_report results/MSL_reference_eval.json \
    --reference_eval_pairs 100 \
    --reference_eval_window_size 100 --reference_eval_stride 100 \
    --reference_eval_guard_size 100 --reference_eval_anomaly_min_fraction 0.50 \
    --model Qwen/Qwen3-4B --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 --perceiver_base_url http://localhost:8001/v1

# SWaT preparation first creates finite, held-out normal windows and preserves
# every source timestamp gap through segment IDs.  Do not evaluate directly on
# the raw CSVs or on a normal+attack concatenation.
python -m annotation.preprocess_swat \
    --normal_csv datasets/SWaT/normal.csv \
    --attack_csv datasets/SWaT/attack.csv \
    --output_dir datasets/SWaT/preprocessed

# SWaT primary framework evaluation (random channels; labels stay hidden).
python -m annotation.run_annotation \
    --reference_eval_test_data datasets/SWaT/preprocessed/SWaT_test.npy \
    --reference_eval_labels datasets/SWaT/preprocessed/SWaT_test_labels.npy \
    --reference_eval_test_segments datasets/SWaT/preprocessed/SWaT_test_segments.npy \
    --reference_eval_reference_data datasets/SWaT/preprocessed/SWaT_reference.npy \
    --reference_eval_reference_segments datasets/SWaT/preprocessed/SWaT_reference_segments.npy \
    --reference_eval_output datasets/SWaT/swat_framework_random.jsonl \
    --reference_eval_report results/SWaT_framework_random.json \
    --reference_eval_pairs 100 --reference_eval_window_size 100 \
    --reference_eval_stride 100 --reference_eval_guard_size 100 \
    --reference_eval_anomaly_min_fraction 0.50 \
    --reference_eval_channel_selection random \
    --dataset_description "Single-channel windows from a real industrial water-treatment telemetry system; assess observable time-series quality for reliable monitoring and downstream forecasting." \
    --model Qwen/Qwen3-4B --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 --perceiver_base_url http://localhost:8001/v1

# WADI uses a separate normal-only reference recording and a labelled test
# recording (1=normal, -1=attack).  Its Row IDs, rather than Date/Time text,
# define continuous segments.
python -m annotation.preprocess_wadi \
    --normal_csv datasets/WADI/WADI_14days_new.csv \
    --attack_csv datasets/WADI/WADI_attackdataLABLE.csv \
    --output_dir datasets/WADI/preprocessed

python -m annotation.run_annotation \
    --reference_eval_test_data datasets/WADI/preprocessed/WADI_test.npy \
    --reference_eval_labels datasets/WADI/preprocessed/WADI_test_labels.npy \
    --reference_eval_test_segments datasets/WADI/preprocessed/WADI_test_segments.npy \
    --reference_eval_reference_data datasets/WADI/preprocessed/WADI_reference.npy \
    --reference_eval_reference_segments datasets/WADI/preprocessed/WADI_reference_segments.npy \
    --reference_eval_output datasets/WADI/wadi_framework_random.jsonl \
    --reference_eval_report results/WADI_framework_random.json \
    --reference_eval_pairs 100 --reference_eval_window_size 100 \
    --reference_eval_stride 100 --reference_eval_guard_size 100 \
    --reference_eval_anomaly_min_fraction 0.50 \
    --reference_eval_channel_selection random \
    --dataset_description "Single-channel windows from a real industrial water-distribution telemetry system; assess observable time-series quality for reliable monitoring and downstream forecasting." \
    --model Qwen/Qwen3-4B --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 --perceiver_base_url http://localhost:8001/v1

# Batch: annotate all datasets listed in a config file
python -m annotation.run_annotation \
    --batch_config annotation/dataset_configs.json \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

from config import Config
from annotation.pairwise_annotator import annotate_pair, is_high_confidence
from annotation.significance import (
    summarize_dimension_significance,
    write_significance_report,
)
from annotation.realworld_evaluation import run_reference_evaluation
from tools import NumpyEncoder

# ── Default targets ────────────────────────────────────────────────────────────
DEFAULT_N_VALID = 500          # target valid pairs per dataset
DEFAULT_OVERSAMPLE = 3          # try up to n_valid × oversample pairs
DEFAULT_MIN_CONFIDENCE = 0.5    # |2p-1| >= this threshold


# ── Data loading ───────────────────────────────────────────────────────────────

def load_blocks(jsonl_path: str) -> list[dict]:
    """Load all records from a blocks.jsonl file."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_existing_pairs(output_path: Path) -> set[tuple[int, int]]:
    """Return set of (block_a, block_b) pairs already written to output."""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["block_a"], rec["block_b"]))
            except Exception:
                pass
    return done


def count_valid(output_path: Path, min_confidence: float) -> int:
    """Count high-confidence pairs already in the output file."""
    if not output_path.exists():
        return 0
    n = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("error") is None and is_high_confidence(
                    rec["comparisons_avg"], min_confidence
                ):
                    n += 1
            except Exception:
                pass
    return n


# ── Sampling ───────────────────────────────────────────────────────────────────

def sample_pairs(
    n_records: int,
    n_valid: int,
    oversample: int,
    seed: int,
    done: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Sample pairs to annotate.

    For small datasets (C(N,2) ≤ n_valid): return all remaining pairs.
    For large datasets: sample oversample × n_valid pairs, minus already done.
    """
    max_possible = n_records * (n_records - 1) // 2
    budget = min(n_valid * oversample, max_possible)

    if max_possible <= n_valid:
        # Small dataset — use all combinations
        all_pairs = [(i, j) for i, j in combinations(range(n_records), 2)
                     if (i, j) not in done]
        random.seed(seed)
        random.shuffle(all_pairs)
        return all_pairs

    # Large dataset — random sample from all combinations
    rng = random.Random(seed)
    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set(done)

    attempts = 0
    max_attempts = budget * 10
    while len(selected) < budget and attempts < max_attempts:
        i = rng.randint(0, n_records - 1)
        j = rng.randint(0, n_records - 1)
        if i == j:
            attempts += 1
            continue
        pair = (min(i, j), max(i, j))
        if pair not in seen:
            seen.add(pair)
            selected.append(pair)
        attempts += 1

    return selected


# ── Single-dataset annotation ──────────────────────────────────────────────────

def annotate_dataset(
    jsonl_path: str,
    output_path: Path,
    dataset_description: str,
    config: Config,
    n_valid: int = DEFAULT_N_VALID,
    oversample: int = DEFAULT_OVERSAMPLE,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    seed: int = 42,
    resume: bool = True,
    save_dimension_results: bool = False,
) -> dict:
    """
    Annotate one dataset. Returns summary stats dict.
    """
    print(f"\n{'='*68}")
    print(f"  Dataset : {jsonl_path}")
    print(f"  Output  : {output_path}")
    print(f"{'='*68}")

    records = load_blocks(jsonl_path)
    n_records = len(records)
    print(f"  Loaded {n_records} blocks")

    max_possible = n_records * (n_records - 1) // 2
    effective_target = min(n_valid, max_possible)
    print(f"  Target valid pairs : {effective_target}  "
          f"(max possible: {max_possible})")

    # Resume: load already done pairs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_pairs = load_existing_pairs(output_path) if resume else set()
    n_valid_existing = count_valid(output_path, min_confidence) if resume else 0

    if n_valid_existing >= effective_target:
        print(f"  Already have {n_valid_existing} valid pairs — skipping.")
        return {"dataset": jsonl_path, "valid": n_valid_existing,
                "total_tried": len(done_pairs), "skipped": True}

    if done_pairs:
        print(f"  Resuming: {len(done_pairs)} pairs already done, "
              f"{n_valid_existing} valid")

    # Sample pairs to try
    pairs = sample_pairs(n_records, effective_target, oversample, seed, done_pairs)
    print(f"  Will try up to {len(pairs)} pairs\n")

    n_tried = 0
    n_valid_new = 0
    n_error = 0

    # Build index lookup
    index_to_record = {r["index"]: r for r in records}
    all_indices = [r["index"] for r in records]

    with open(output_path, "a", encoding="utf-8") as fout:
        for pos_a, pos_b in pairs:
            if n_valid_existing + n_valid_new >= effective_target:
                print(f"\n  Target reached ({effective_target} valid pairs).")
                break

            idx_a = all_indices[pos_a]
            idx_b = all_indices[pos_b]
            rec_a = index_to_record[idx_a]
            rec_b = index_to_record[idx_b]

            t0 = time.time()
            result = annotate_pair(
                index_a=idx_a,
                series_a=rec_a["input_arr"],
                index_b=idx_b,
                series_b=rec_b["input_arr"],
                dataset_description=dataset_description,
                config=config,
                save_dimension_results=save_dimension_results,
            )
            elapsed = time.time() - t0

            n_tried += 1

            if result["error"]:
                n_error += 1
                print(f"  [{n_tried}] ERROR: {result['error'][:80]}")
            else:
                valid = is_high_confidence(result["comparisons_avg"], min_confidence)
                if valid:
                    n_valid_new += 1

                status = "✓" if valid else "~"
                print(
                    f"  [{n_tried}]  {status}  "
                    f"({idx_a},{idx_b})  "
                    f"winner={result['winner']}  conf={result['confidence']:.2f}  "
                    f"avg={result['comparisons_avg']:.3f}  "
                    f"[{elapsed:.1f}s]  "
                    f"valid={n_valid_existing + n_valid_new}/{effective_target}"
                )

            fout.write(json.dumps(result, ensure_ascii=False, cls=NumpyEncoder) + "\n")
            fout.flush()

    total_valid = n_valid_existing + n_valid_new
    print(f"\n  Done.  tried={n_tried}  valid={total_valid}  errors={n_error}")
    return {
        "dataset": jsonl_path,
        "valid": total_valid,
        "total_tried": len(done_pairs) + n_tried,
        "errors": n_error,
        "skipped": False,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise annotation for TSqualityAgent"
    )

    # Input / output
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to blocks.jsonl for a single dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (single dataset mode)")
    parser.add_argument("--dataset_description", type=str, default="",
                        help="Natural-language description of the dataset "
                             "(passed to agent as context)")
    parser.add_argument("--batch_config", type=str, default=None,
                        help="JSON file listing multiple datasets. Each entry: "
                             "{\"dataset\": ..., \"output\": ..., \"description\": ...}")

    # Annotation parameters
    parser.add_argument("--n_valid", type=int, default=DEFAULT_N_VALID,
                        help=f"Target valid pairs per dataset (default {DEFAULT_N_VALID})")
    parser.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE,
                        help="Try up to n_valid × oversample pairs to reach target "
                             f"(default {DEFAULT_OVERSAMPLE})")
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help="Minimum |2p-1| for a pair to count as valid "
                             f"(default {DEFAULT_MIN_CONFIDENCE})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip pairs already written to output (default: True)")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                        help="Restart from scratch (ignore existing output)")
    parser.add_argument("--save_dimension_results", action="store_true",
                        help="Persist compact Perceiver/Inspector results in each JSONL record. "
                             "Required for a later significance report; enabled automatically "
                             "when --significance_report is set.")
    parser.add_argument("--significance_report", type=str, default=None,
                        help="Write a JSON and CSV Perceiver dimension-selection count after annotation "
                            "(single-dataset mode only).")

    # Real telemetry evaluation: external anomaly labels are never passed to Agent.
    parser.add_argument("--reference_eval_test_data", type=str, default=None,
                        help="[time, channels] .npy test matrix for real-telemetry evaluation")
    parser.add_argument("--reference_eval_labels", type=str, default=None,
                        help="Boolean/0-1 .npy anomaly labels aligned to test timestamps")
    parser.add_argument("--reference_eval_reference_data", type=str, default=None,
                        help="Normal-only [time, channels] .npy reference matrix")
    parser.add_argument("--reference_eval_test_segments", type=str, default=None,
                        help="Optional .npy segment IDs; no evaluation window may cross a segment boundary")
    parser.add_argument("--reference_eval_reference_segments", type=str, default=None,
                        help="Optional .npy segment IDs; no reference window may cross a segment boundary")
    parser.add_argument("--reference_eval_output", type=str, default=None,
                        help="JSONL output for pair-level real-telemetry evaluation records")
    parser.add_argument("--reference_eval_report", type=str, default=None,
                        help="JSON aggregate report for real-telemetry evaluation")
    parser.add_argument("--reference_eval_pairs", type=int, default=100,
                        help="Number of normal-anomaly pairs (default: 100)")
    parser.add_argument("--reference_eval_window_size", type=int, default=100)
    parser.add_argument("--reference_eval_stride", type=int, default=100)
    parser.add_argument("--reference_eval_guard_size", type=int, default=100,
                        help="Normal windows must have no anomaly within this many adjacent points")
    parser.add_argument("--reference_eval_anomaly_min_fraction", type=float, default=0.5,
                        help="Minimum labelled-anomaly fraction in an anomaly window (default: 0.5)")
    parser.add_argument("--reference_eval_reference_stride", type=int, default=None,
                        help="Stride for normal reference windows (default: window size)")
    parser.add_argument("--reference_eval_label_quantile", type=float, default=0.95,
                        help="Normal-normal score-gap quantile defining a relevant weak label")
    parser.add_argument("--reference_eval_channel_selection", type=str, default="random",
                        choices=["random", "max_reference_evidence"],
                        help="random for the primary temporal-label evaluation; "
                             "max_reference_evidence only for the diagnostic subagent study")

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="Main model for Inspector + Adjudicator")
    parser.add_argument("--base_url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--perceiver_model", type=str, default="",
                        help="Model alias for Perceiver (e.g. perceiver-grpo-v2). "
                             "Falls back to --model if empty.")
    parser.add_argument("--perceiver_base_url", type=str, default="",
                        help="Base URL for Perceiver model. Falls back to --base_url.")
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=6,
                        help="Max ReAct steps per quality dimension in Inspector")
    parser.add_argument("--max_recheck", type=int, default=2,
                        help="Max times Adjudicator sends Inspector back for recheck")
    parser.add_argument("--max_replan", type=int, default=1,
                        help="Max times Adjudicator sends Perceiver back for replanning")

    args = parser.parse_args()

    config = Config(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        enable_thinking=args.enable_thinking,
        max_steps_per_dimension=args.max_steps,
        max_recheck=args.max_recheck,
        max_replan=args.max_replan,
        perceiver_model=args.perceiver_model,
        perceiver_base_url=args.perceiver_base_url,
    )

    reference_mode_values = (
        args.reference_eval_test_data,
        args.reference_eval_labels,
        args.reference_eval_reference_data,
        args.reference_eval_output,
        args.reference_eval_report,
    )
    if any(value is not None for value in reference_mode_values):
        if not all(value is not None for value in reference_mode_values):
            parser.error(
                "Real-telemetry mode requires --reference_eval_test_data, --reference_eval_labels, "
                "--reference_eval_reference_data, --reference_eval_output, and --reference_eval_report."
            )
        if args.dataset or args.batch_config:
            parser.error("Real-telemetry mode cannot be combined with --dataset or --batch_config.")
        report = run_reference_evaluation(
            test_data_path=args.reference_eval_test_data,
            labels_path=args.reference_eval_labels,
            reference_data_path=args.reference_eval_reference_data,
            test_segments_path=args.reference_eval_test_segments,
            reference_segments_path=args.reference_eval_reference_segments,
            output_path=args.reference_eval_output,
            report_path=args.reference_eval_report,
            config=config,
            n_pairs=args.reference_eval_pairs,
            window_size=args.reference_eval_window_size,
            stride=args.reference_eval_stride,
            normal_guard_size=args.reference_eval_guard_size,
            anomaly_min_fraction=args.reference_eval_anomaly_min_fraction,
            reference_stride=args.reference_eval_reference_stride,
            label_quantile=args.reference_eval_label_quantile,
            channel_selection=args.reference_eval_channel_selection,
            seed=args.seed,
            dataset_description=args.dataset_description,
            resume=args.resume,
        )
        framework = report["results"]["full_framework"]
        perceiver = report["results"]["perceiver_vs_reference_weak_labels"]
        inspector = report["results"]["targeted_inspector_vs_reference_weak_labels"]
        print(f"\n  Real-telemetry report: {args.reference_eval_report}")
        print(
            "  Full framework normal-preference score (tie=0.5): "
            f"{framework['normal_preference_score_tie_half']}"
        )
        print(f"  Perceiver weak-label micro-F1: {perceiver['micro_f1']}")
        print(f"  Targeted Inspector weak-label accuracy: {inspector['winner_accuracy']}")
        return

    # ── Collect standard annotation tasks ──────────────────────────────────────
    tasks: list[dict] = []

    if args.batch_config:
        with open(args.batch_config, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        print(f"Batch mode: {len(tasks)} datasets from {args.batch_config}")
    elif args.dataset:
        if not args.output:
            args.output = str(Path(args.dataset).parent / "annotation.jsonl")
        tasks = [{
            "dataset": args.dataset,
            "output": args.output,
            "description": args.dataset_description,
        }]
    else:
        parser.error("Provide either --dataset or --batch_config")

    if args.significance_report and len(tasks) != 1:
        parser.error("--significance_report is supported only with a single --dataset")

    # ── Run ────────────────────────────────────────────────────────────────────
    summaries = []
    for task in tasks:
        summary = annotate_dataset(
            jsonl_path=task["dataset"],
            output_path=Path(task["output"]),
            dataset_description=task.get("description", ""),
            config=config,
            n_valid=args.n_valid,
            oversample=args.oversample,
            min_confidence=args.min_confidence,
            seed=args.seed,
            resume=args.resume,
            save_dimension_results=(args.save_dimension_results or bool(args.significance_report)),
        )
        summaries.append(summary)

    if args.significance_report:
        report_path = Path(args.significance_report)
        annotation_path = Path(tasks[0]["output"])
        try:
            report = summarize_dimension_significance(
                load_blocks(str(annotation_path)),
            )
        except ValueError as exc:
            parser.error(str(exc))
        csv_path = write_significance_report(report, report_path)
        print(f"\n  Significance report: {report_path}")
        print(f"  Significance table : {csv_path}")
        print(f"  Pairwise comparisons: {report['n_pairwise_comparisons']}")
        print("  Top dimensions:")
        for row in report["dimensions"][:3]:
            print(f"    {row['dimension']}: significance={row['significance']}")

    # ── Final summary ──────────────────────────────────────────────────────────
    if len(summaries) > 1:
        print(f"\n{'='*68}")
        print(f"  Batch complete — {len(summaries)} datasets")
        print(f"  {'Dataset':<40}  {'Valid':>8}  {'Tried':>8}")
        print(f"  {'-'*40}  {'-'*8}  {'-'*8}")
        for s in summaries:
            name = Path(s["dataset"]).parent.name
            print(f"  {name:<40}  {s['valid']:>8}  {s['total_tried']:>8}")
        total_valid = sum(s["valid"] for s in summaries)
        print(f"\n  Total valid pairs: {total_valid}")


if __name__ == "__main__":
    main()
