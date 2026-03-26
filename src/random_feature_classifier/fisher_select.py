from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .cache import load_feature_cache
from .selection import fisher_scores, rank_features, save_selection, topk_with_correlation_pruning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank cached features by multiclass Fisher score.")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--max-ranked-features", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-abs-correlation", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_features, train_labels, metadata = load_feature_cache(args.train_cache)
    scores = fisher_scores(train_features, train_labels, args.num_classes)
    ranked_indices = rank_features(scores)

    if args.max_ranked_features > 0:
        ranked_indices = ranked_indices[: args.max_ranked_features]

    if args.top_k > 0:
        ranked_indices = topk_with_correlation_pruning(
            train_features,
            ranked_indices,
            top_k=args.top_k,
            max_abs_correlation=args.max_abs_correlation,
        )

    output_path = Path(args.output_path)
    save_selection(
        output_path,
        scores=scores,
        ranked_indices=ranked_indices,
        metadata={
            "train_cache": args.train_cache,
            "num_classes": args.num_classes,
            "max_ranked_features": args.max_ranked_features,
            "top_k": args.top_k,
            "max_abs_correlation": args.max_abs_correlation,
            "feature_dim": train_features.shape[1],
            "backbone_kind": metadata.get("backbone_kind"),
            "tap_stages": metadata.get("tap_stages"),
            "global_feature_dim": metadata.get("global_feature_dim"),
            "width_multiplier": metadata.get("width_multiplier"),
            "seed": metadata.get("seed"),
        },
    )
    print(
        f"saved selection path={output_path} "
        f"ranked_count={ranked_indices.numel()} "
        f"best_score={float(scores[ranked_indices[0]].item()):.6f}"
    )


if __name__ == "__main__":
    main()
