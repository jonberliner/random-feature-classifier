from pathlib import Path

import torch

from random_feature_classifier.selection import (
    fisher_scores,
    load_selection,
    one_vs_rest_fisher_scores,
    rank_features,
    save_selection,
    topk_with_correlation_pruning,
)


def test_fisher_ranks_discriminative_feature_first() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    features = torch.tensor(
        [
            [0.0, 1.0, 0.1],
            [0.0, 1.1, 0.0],
            [5.0, 0.9, 0.1],
            [5.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    scores = fisher_scores(features, labels, num_classes=2)
    ranked = rank_features(scores)
    assert int(ranked[0].item()) == 0


def test_selection_round_trip_and_correlation_pruning(tmp_path: Path) -> None:
    base = torch.linspace(-1.0, 1.0, 10)
    features = torch.stack((base, base * 0.99, torch.randn(10)), dim=1)
    ranked = torch.tensor([0, 1, 2], dtype=torch.int64)
    pruned = topk_with_correlation_pruning(features, ranked, top_k=2, max_abs_correlation=0.9)
    assert torch.equal(pruned, torch.tensor([0, 2], dtype=torch.int64))

    path = tmp_path / "selection.pt"
    save_selection(path, torch.tensor([3.0, 2.0, 1.0]), pruned, {"name": "test"})
    payload = load_selection(str(path))
    assert payload["name"] == "test"
    assert torch.equal(payload["ranked_indices"], pruned)


def test_one_vs_rest_fisher_scores_capture_class_specific_feature() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    features = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [2.8, 0.1, 0.0],
            [0.0, 3.0, 0.0],
            [0.1, 2.9, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 2.8],
        ],
        dtype=torch.float32,
    )
    scores = one_vs_rest_fisher_scores(features, labels, num_classes=3)
    assert int(scores[0].argmax().item()) == 0
    assert int(scores[1].argmax().item()) == 1
    assert int(scores[2].argmax().item()) == 2
