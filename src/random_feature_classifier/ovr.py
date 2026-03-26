from __future__ import annotations

import torch


def one_vs_rest_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=num_classes).to(torch.float32)


def refined_scores_with_ovr(
    coarse_logits: torch.Tensor,
    ovr_logits: torch.Tensor,
    top_k: int,
    residual_scale: float,
    margin_threshold: float,
) -> torch.Tensor:
    if top_k < 1 or residual_scale == 0.0:
        return coarse_logits

    candidate_count = min(top_k, coarse_logits.shape[1])
    candidate_scores, candidate_classes = torch.topk(coarse_logits, k=candidate_count, dim=1)
    updates = residual_scale * ovr_logits.gather(1, candidate_classes)

    if margin_threshold >= 0.0 and candidate_count > 1:
        margins = candidate_scores[:, 0] - candidate_scores[:, 1]
        gate = (margins < margin_threshold).to(dtype=updates.dtype).unsqueeze(1)
        updates = updates * gate

    refined = coarse_logits.clone()
    refined.scatter_add_(1, candidate_classes, updates)
    return refined


def refine_with_ovr_predictions(
    coarse_logits: torch.Tensor,
    ovr_logits: torch.Tensor,
    top_k: int,
    residual_scale: float,
    margin_threshold: float,
) -> torch.Tensor:
    return refined_scores_with_ovr(
        coarse_logits,
        ovr_logits,
        top_k=top_k,
        residual_scale=residual_scale,
        margin_threshold=margin_threshold,
    ).argmax(dim=1)
