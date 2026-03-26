import torch

from random_feature_classifier.pairwise import (
    class_pairs,
    pairwise_targets,
    pairwise_vote_predictions,
    pairwise_vote_tallies,
    topk_refine_predictions,
)


def test_class_pairs_count_matches_one_vs_one() -> None:
    pairs = class_pairs(10)
    assert len(pairs) == 45
    assert pairs[0] == (0, 1)
    assert pairs[-1] == (8, 9)


def test_pairwise_targets_only_activate_relevant_pairs() -> None:
    labels = torch.tensor([0, 2], dtype=torch.int64)
    targets, mask = pairwise_targets(labels, num_classes=4)
    pairs = class_pairs(4)

    assert int(mask[0].sum().item()) == 3
    assert int(mask[1].sum().item()) == 3
    assert float(targets[0, pairs.index((0, 1))].item()) == 0.0
    assert float(targets[0, pairs.index((0, 2))].item()) == 0.0
    assert float(targets[1, pairs.index((0, 2))].item()) == 1.0
    assert float(mask[0, pairs.index((1, 2))].item()) == 0.0


def test_pairwise_vote_prediction_uses_margin_tie_break() -> None:
    logits = torch.tensor(
        [
            [1.0, 2.0, 1.5],
            [-1.0, -0.5, 3.0],
        ],
        dtype=torch.float32,
    )
    wins, _ = pairwise_vote_tallies(logits, num_classes=3)
    predictions = pairwise_vote_predictions(logits, num_classes=3)

    assert torch.equal(wins[0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(wins[1], torch.tensor([2.0, 0.0, 1.0]))
    assert torch.equal(predictions, torch.tensor([2, 0], dtype=torch.int64))


def test_topk_refine_predictions_only_adjusts_top_candidates() -> None:
    global_logits = torch.tensor([[3.0, 2.9, 2.8, 0.5]], dtype=torch.float32)
    pairwise_logits = torch.zeros((1, len(class_pairs(4))), dtype=torch.float32)
    pair_lookup = {pair: index for index, pair in enumerate(class_pairs(4))}
    pairwise_logits[0, pair_lookup[(0, 1)]] = 1.0
    pairwise_logits[0, pair_lookup[(0, 2)]] = 1.0
    pairwise_logits[0, pair_lookup[(1, 2)]] = 1.0

    predictions = topk_refine_predictions(
        global_logits,
        pairwise_logits,
        num_classes=4,
        top_k=3,
        pairwise_scale=0.5,
    )
    assert torch.equal(predictions, torch.tensor([2], dtype=torch.int64))
