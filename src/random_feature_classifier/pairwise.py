from __future__ import annotations

import torch


def class_pairs(num_classes: int) -> list[tuple[int, int]]:
    return [(left, right) for left in range(num_classes) for right in range(left + 1, num_classes)]


def pair_index_lookup(num_classes: int) -> dict[tuple[int, int], int]:
    return {pair: index for index, pair in enumerate(class_pairs(num_classes))}


def pair_index_tensor(num_classes: int, device: torch.device) -> torch.Tensor:
    lookup = torch.full((num_classes, num_classes), -1, dtype=torch.int64, device=device)
    for pair_index, (left, right) in enumerate(class_pairs(num_classes)):
        lookup[left, right] = pair_index
    return lookup


def pair_offset_tensors(width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    left_offsets = []
    right_offsets = []
    for left in range(width):
        for right in range(left + 1, width):
            left_offsets.append(left)
            right_offsets.append(right)
    return (
        torch.tensor(left_offsets, dtype=torch.int64, device=device),
        torch.tensor(right_offsets, dtype=torch.int64, device=device),
    )


def pairwise_targets(labels: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = class_pairs(num_classes)
    batch_size = labels.shape[0]
    targets = torch.zeros((batch_size, len(pairs)), dtype=torch.float32)
    mask = torch.zeros((batch_size, len(pairs)), dtype=torch.float32)

    for pair_index, (left, right) in enumerate(pairs):
        left_mask = labels == left
        right_mask = labels == right
        valid = left_mask | right_mask
        mask[valid, pair_index] = 1.0
        targets[right_mask, pair_index] = 1.0

    return targets, mask


def pairwise_vote_tallies(logits: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = class_pairs(num_classes)
    wins = torch.zeros((logits.shape[0], num_classes), device=logits.device, dtype=torch.float32)
    margins = torch.zeros((logits.shape[0], num_classes), device=logits.device, dtype=torch.float32)

    for pair_index, (left, right) in enumerate(pairs):
        margin = logits[:, pair_index]
        right_wins = margin > 0.0
        left_wins = ~right_wins
        wins[:, left] += left_wins.to(torch.float32)
        wins[:, right] += right_wins.to(torch.float32)
        margins[:, left] += -margin
        margins[:, right] += margin

    return wins, margins


def pairwise_vote_predictions(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    wins, margins = pairwise_vote_tallies(logits, num_classes)
    max_wins = wins.amax(dim=1, keepdim=True)
    tied = wins == max_wins
    tie_break_scores = torch.where(tied, margins, torch.full_like(margins, float("-inf")))
    return tie_break_scores.argmax(dim=1)


def topk_refine_predictions(
    global_logits: torch.Tensor,
    pairwise_logits: torch.Tensor,
    num_classes: int,
    top_k: int,
    pairwise_scale: float,
) -> torch.Tensor:
    if top_k < 2:
        return global_logits.argmax(dim=1)

    refined_scores = global_logits.clone()
    candidate_classes = torch.topk(global_logits, k=min(top_k, num_classes), dim=1).indices
    if candidate_classes.shape[1] < 2:
        return refined_scores.argmax(dim=1)

    left_offsets, right_offsets = pair_offset_tensors(candidate_classes.shape[1], device=global_logits.device)
    left_classes = candidate_classes[:, left_offsets]
    right_classes = candidate_classes[:, right_offsets]
    lower = torch.minimum(left_classes, right_classes)
    upper = torch.maximum(left_classes, right_classes)

    lookup = pair_index_tensor(num_classes, device=global_logits.device)
    pair_indices = lookup[lower, upper]
    margins = pairwise_logits.gather(1, pair_indices)
    scaled_margins = pairwise_scale * margins
    left_delta = torch.where(left_classes < right_classes, -scaled_margins, scaled_margins)
    right_delta = -left_delta

    refined_scores.scatter_add_(1, left_classes, left_delta)
    refined_scores.scatter_add_(1, right_classes, right_delta)

    return refined_scores.argmax(dim=1)
