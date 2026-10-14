"""
Inference: compute scalar quality scores for all series in a dataset.

Given a trained ScoreModel checkpoint and a blocks.jsonl, this module:
  1. Computes MOMENT embeddings for all series
  2. (Optionally) few-shot fine-tunes the model on a small annotation subset
  3. Runs forward pass to produce a quality score for each series index
  4. Saves scores to a JSONL file: {"index": int, "quality_score": float}

Usage
-----
# Single dataset
python -m meta_learning_rater.score \
    --blocks datasets/electricity/blocks.jsonl \
    --model  meta_learning_rater/checkpoints/tsrater.pth \
    --annotation datasets/electricity/annotation.jsonl

# For a per-dataset rater, replace --model with that dataset's checkpoint (e.g., rater_electricity.pth).

# Batch: score all datasets in dataset_configs.json
python -m meta_learning_rater.score \
    --config annotation/dataset_configs.json \
    --model  meta_learning_rater/checkpoints/tsrater.pth
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from meta_learning_rater.dataset import (
    load_blocks,
    load_annotation,
    load_moment_model,
    compute_embeddings,
    PairwiseDataset,
)
from meta_learning_rater.model import ScoreModel, BradleyTerryLoss


def _synchronize_cuda_if_needed(device: str) -> None:
    """Flush queued CUDA work so adaptation timing includes GPU execution."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_score_model(
    checkpoint_path: str | Path,
    input_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    device: str = "cpu",
) -> ScoreModel:
    model = ScoreModel(input_dim, hidden_dim, num_layers)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # New per-dataset checkpoints store the normalisation fitted during
    # training.  Older meta-rater checkpoints are plain state_dict objects and
    # deliberately remain supported.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        normalization = checkpoint.get("embedding_normalization")
    else:
        state = checkpoint
        normalization = None
    model.load_state_dict(state)
    model.embedding_normalization = normalization
    return model.to(device)


def _normalize_embeddings_for_checkpoint(
    embeddings: dict[int, torch.Tensor],
    model: ScoreModel,
) -> dict[int, torch.Tensor]:
    """Apply the training-time embedding normalisation, if the checkpoint has it."""
    normalization = getattr(model, "embedding_normalization", None)
    if not normalization:
        return embeddings
    mean = torch.as_tensor(normalization["mean"], dtype=torch.float32, device="cpu")
    std = torch.as_tensor(normalization["std"], dtype=torch.float32, device="cpu").clamp(min=1e-6)
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError("Checkpoint embedding_normalization has invalid shapes.")
    print("Applying checkpoint embedding normalization …")
    return {index: (embedding - mean) / std for index, embedding in embeddings.items()}


def few_shot_adapt(
    model: ScoreModel,
    pairs: list[dict],
    embeddings: dict[int, torch.Tensor],
    adaptation_steps: int = 10,
    adaptation_lr: float = 5e-3,
    few_shot_n: int = 20,
    device: str = "cpu",
) -> tuple[ScoreModel, float]:
    """Fine-tune on a few pairs and return the pure update-loop duration."""
    import copy
    from torch.utils.data import DataLoader

    model = copy.deepcopy(model).to(device)
    dataset = PairwiseDataset(embeddings, pairs[:few_shot_n])
    if len(dataset) == 0:
        return model, 0.0

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=adaptation_lr)
    loss_fn = BradleyTerryLoss()
    model.train()

    # CUDA is asynchronous; synchronize to measure only the gradient updates.
    _synchronize_cuda_if_needed(device)
    adaptation_start = time.perf_counter()
    for _ in range(adaptation_steps):
        for emb_a, emb_b, p in loader:
            emb_a = emb_a.to(device)
            emb_b = emb_b.to(device)
            p     = p.to(device)
            loss = loss_fn(model(emb_a), model(emb_b), p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    _synchronize_cuda_if_needed(device)

    return model, time.perf_counter() - adaptation_start


def score_dataset(
    blocks_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    annotation_path: str | Path | None = None,
    hidden_dim: int = 256,
    num_layers: int = 3,
    adaptation_steps: int = 10,
    adaptation_lr: float = 5e-3,
    few_shot_n: int = 20,
    device: str | None = None,
    score_batch_size: int = 1024,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading MOMENT encoder …")
    moment = load_moment_model()

    print(f"Computing embeddings for {blocks_path} …")
    blocks = load_blocks(blocks_path)
    embeddings = compute_embeddings(moment, blocks, device=device)

    # Infer input_dim from first embedding
    input_dim = next(iter(embeddings.values())).shape[0]

    print(f"Loading ScoreModel from {checkpoint_path} …")
    model = load_score_model(checkpoint_path, input_dim, hidden_dim, num_layers, device)
    embeddings = _normalize_embeddings_for_checkpoint(embeddings, model)

    # Optional: few-shot adapt on this dataset's annotation
    if annotation_path and Path(annotation_path).exists():
        pairs = load_annotation(annotation_path)
        if pairs:
            print(f"Few-shot adapting on {min(few_shot_n, len(pairs))} pairs …")
            model, adaptation_elapsed = few_shot_adapt(
                model, pairs, embeddings,
                adaptation_steps=adaptation_steps,
                adaptation_lr=adaptation_lr,
                few_shot_n=few_shot_n,
                device=device,
            )
            print(f"Few-shot adaptation loop time: {adaptation_elapsed:.2f} s "
                  f"({adaptation_elapsed / 60:.2f} min)")

    # Score all series
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive.")
    all_indices = [r["index"] for r in blocks]
    n_scored = 0
    with torch.no_grad():
        with open(output_path, "w", encoding="utf-8") as f:
            for start in range(0, len(all_indices), score_batch_size):
                batch_indices = all_indices[start:start + score_batch_size]
                batch_embs = torch.stack([embeddings[index] for index in batch_indices]).to(device)
                scores = model(batch_embs).cpu().tolist()
                for index, score in zip(batch_indices, scores):
                    f.write(json.dumps({"index": index, "quality_score": score}) + "\n")
                    n_scored += 1

    print(f"Scores saved → {output_path}  ({n_scored} records)")


def score_all_datasets(
    config_path: str | Path,
    checkpoint_path: str | Path,
    hidden_dim: int = 256,
    num_layers: int = 3,
    adaptation_steps: int = 10,
    adaptation_lr: float = 5e-3,
    few_shot_n: int = 20,
    device: str | None = None,
    score_batch_size: int = 1024,
) -> None:
    """Score all datasets listed in a dataset_configs.json.

    Output is written to datasets/<name>/scores.jsonl alongside blocks.jsonl.
    Uses annotation.jsonl from the same directory for few-shot adaptation if present.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    print(f"Scoring {len(configs)} datasets from {config_path}\n")
    for cfg in configs:
        blocks_path     = Path(cfg["dataset"])
        annotation_path = Path(cfg["output"])
        output_path     = blocks_path.parent / "scores.jsonl"
        name            = blocks_path.parent.name

        if not blocks_path.exists():
            print(f"  [skip] {name}: blocks.jsonl not found")
            continue

        print(f"  [{name}]")
        score_dataset(
            blocks_path=blocks_path,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            annotation_path=annotation_path if annotation_path.exists() else None,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            adaptation_steps=adaptation_steps,
            adaptation_lr=adaptation_lr,
            few_shot_n=few_shot_n,
            device=device,
            score_batch_size=score_batch_size,
        )

    print("\nAll datasets scored.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score time-series quality")

    # Mode: single dataset or batch
    parser.add_argument("--blocks",  default=None, help="blocks.jsonl path (single dataset mode)")
    parser.add_argument("--output",  default=None, help="Output scores.jsonl (single dataset mode)")
    parser.add_argument("--config",  default=None,
                        help="dataset_configs.json path (batch mode — scores all datasets)")

    parser.add_argument("--model",       default=None,
                        help="ScoreModel .pth checkpoint (single shared model, used for meta-learning rater)")
    parser.add_argument("--model_dir",   default=None,
                        help="Directory containing per-dataset checkpoints named rater_<name>.pth "
                             "(used for single-dataset rater, batch mode only)")
    parser.add_argument("--annotation",  default=None,  help="annotation.jsonl for few-shot adapt (single mode)")
    parser.add_argument("--hidden_dim",  type=int, default=256)
    parser.add_argument("--num_layers",  type=int, default=3)
    parser.add_argument("--adapt_steps", type=int, default=10)
    parser.add_argument("--adapt_lr",    type=float, default=5e-3)
    parser.add_argument("--few_shot_n",  type=int, default=20)
    parser.add_argument("--score_batch_size", type=int, default=1024,
                        help="Number of embeddings per rater forward pass")
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    if not args.model and not args.model_dir:
        parser.error("Provide either --model (shared checkpoint) or --model_dir (per-dataset checkpoints)")

    if args.config:
        if args.model_dir:
            # Per-dataset checkpoints: rater_<name>.pth
            with open(args.config, "r", encoding="utf-8") as f:
                configs = json.load(f)
            model_dir = Path(args.model_dir)
            for cfg in configs:
                blocks_path     = Path(cfg["dataset"])
                annotation_path = Path(cfg["output"])
                name            = blocks_path.parent.name
                ckpt            = model_dir / f"rater_{name}.pth"
                if not ckpt.exists():
                    print(f"  [skip] {name}: checkpoint not found ({ckpt})")
                    continue
                output_path = blocks_path.parent / "scores.jsonl"
                print(f"  [{name}]")
                score_dataset(
                    blocks_path=blocks_path,
                    checkpoint_path=ckpt,
                    output_path=output_path,
                    annotation_path=annotation_path if annotation_path.exists() else None,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    adaptation_steps=args.adapt_steps,
                    adaptation_lr=args.adapt_lr,
                    few_shot_n=args.few_shot_n,
                    device=args.device,
                    score_batch_size=args.score_batch_size,
                )
        else:
            score_all_datasets(
                config_path=args.config,
                checkpoint_path=args.model,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                adaptation_steps=args.adapt_steps,
                adaptation_lr=args.adapt_lr,
                few_shot_n=args.few_shot_n,
                device=args.device,
                score_batch_size=args.score_batch_size,
            )
    elif args.blocks:
        if not args.model:
            parser.error("--model is required for single dataset mode")
        if not args.output:
            args.output = str(Path(args.blocks).parent / "scores.jsonl")
        score_dataset(
            blocks_path=args.blocks,
            checkpoint_path=args.model,
            output_path=args.output,
            annotation_path=args.annotation,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            adaptation_steps=args.adapt_steps,
            adaptation_lr=args.adapt_lr,
            few_shot_n=args.few_shot_n,
            device=args.device,
            score_batch_size=args.score_batch_size,
        )
    else:
        parser.error("Provide either --blocks (single dataset) or --config (batch mode)")


if __name__ == "__main__":
    main()
