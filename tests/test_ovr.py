import torch

from random_feature_classifier.ovr import one_vs_rest_targets, refine_with_ovr_predictions


def test_one_vs_rest_targets_match_one_hot() -> None:
    labels = torch.tensor([0, 2, 1], dtype=torch.int64)
    targets = one_vs_rest_targets(labels, num_classes=4)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(targets, expected)


def test_refine_with_ovr_predictions_updates_topk_classes_only() -> None:
    coarse_logits = torch.tensor([[3.0, 2.9, 2.8, 0.1]], dtype=torch.float32)
    residual_logits = torch.tensor([[-2.0, -1.0, 3.0, 10.0]], dtype=torch.float32)
    predictions = refine_with_ovr_predictions(
        coarse_logits,
        residual_logits,
        top_k=3,
        residual_scale=0.5,
        margin_threshold=-1.0,
    )
    assert torch.equal(predictions, torch.tensor([2], dtype=torch.int64))


def test_refine_with_ovr_predictions_respects_margin_threshold() -> None:
    coarse_logits = torch.tensor(
        [
            [3.0, 2.0, 0.5],
            [3.0, 2.9, 0.5],
        ],
        dtype=torch.float32,
    )
    residual_logits = torch.tensor(
        [
            [-3.0, 4.0, 0.0],
            [-3.0, 4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    predictions = refine_with_ovr_predictions(
        coarse_logits,
        residual_logits,
        top_k=2,
        residual_scale=0.5,
        margin_threshold=0.5,
    )
    assert torch.equal(predictions, torch.tensor([0, 1], dtype=torch.int64))
