from __future__ import annotations

from pathlib import Path

import torch


def class_statistics(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = features.to(torch.float64)
    label_vector = labels.to(torch.int64)
    feature_dim = matrix.shape[1]
    means = torch.zeros((num_classes, feature_dim), dtype=torch.float64)
    variances = torch.zeros((num_classes, feature_dim), dtype=torch.float64)
    counts = torch.zeros((num_classes,), dtype=torch.float64)

    for class_index in range(num_classes):
        mask = label_vector == class_index
        class_count = int(mask.sum().item())
        if class_count == 0:
            continue
        class_features = matrix[mask]
        class_mean = class_features.mean(dim=0)
        centered = class_features - class_mean
        means[class_index] = class_mean
        variances[class_index] = centered.square().mean(dim=0)
        counts[class_index] = float(class_count)

    return means, variances, counts


def fisher_scores(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = features.to(torch.float64)
    label_vector = labels.to(torch.int64)
    global_mean = matrix.mean(dim=0)
    between = torch.zeros_like(global_mean)
    within = torch.zeros_like(global_mean)

    for class_index in range(num_classes):
        mask = label_vector == class_index
        class_count = int(mask.sum().item())
        if class_count == 0:
            continue
        class_features = matrix[mask]
        class_mean = class_features.mean(dim=0)
        centered = class_features - class_mean
        class_var = centered.square().mean(dim=0)
        between = between + class_count * (class_mean - global_mean).square()
        within = within + class_count * class_var

    scores = between / (within + 1e-12)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores.to(torch.float32)


def pairwise_fisher_scores(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    means, variances, counts = class_statistics(features, labels, num_classes)
    pair_scores = []

    for class_a in range(num_classes):
        if counts[class_a] == 0:
            continue
        for class_b in range(class_a + 1, num_classes):
            if counts[class_b] == 0:
                continue
            numerator = (means[class_a] - means[class_b]).square()
            denominator = variances[class_a] + variances[class_b] + 1e-12
            pair_scores.append(numerator / denominator)

    if not pair_scores:
        return torch.zeros((0, features.shape[1]), dtype=torch.float32)
    stacked = torch.stack(pair_scores, dim=0)
    stacked = torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    return stacked.to(torch.float32)


def one_vs_rest_fisher_scores(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = features.to(torch.float64)
    label_vector = labels.to(torch.int64)
    feature_dim = matrix.shape[1]
    scores = []

    for class_index in range(num_classes):
        positive_mask = label_vector == class_index
        negative_mask = ~positive_mask
        positive_count = int(positive_mask.sum().item())
        negative_count = int(negative_mask.sum().item())
        if positive_count == 0 or negative_count == 0:
            scores.append(torch.zeros((feature_dim,), dtype=torch.float64))
            continue
        positive_features = matrix[positive_mask]
        negative_features = matrix[negative_mask]
        positive_mean = positive_features.mean(dim=0)
        negative_mean = negative_features.mean(dim=0)
        positive_var = (positive_features - positive_mean).square().mean(dim=0)
        negative_var = (negative_features - negative_mean).square().mean(dim=0)
        numerator = (positive_mean - negative_mean).square()
        denominator = positive_var + negative_var + 1e-12
        scores.append(numerator / denominator)

    stacked = torch.stack(scores, dim=0)
    stacked = torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    return stacked.to(torch.float32)


def pairwise_fisher_summary(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairwise = pairwise_fisher_scores(features, labels, num_classes)
    if pairwise.shape[0] == 0:
        zeros = torch.zeros((features.shape[1],), dtype=torch.float32)
        return zeros, zeros, zeros

    pairwise_mean = pairwise.mean(dim=0)
    pairwise_max = pairwise.amax(dim=0)
    summary = pairwise_mean + alpha * pairwise_max
    return summary, pairwise_mean, pairwise_max


def one_vs_rest_fisher_summary(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    one_vs_rest = one_vs_rest_fisher_scores(features, labels, num_classes)
    if one_vs_rest.shape[0] == 0:
        zeros = torch.zeros((features.shape[1],), dtype=torch.float32)
        return zeros, zeros, zeros

    ovr_mean = one_vs_rest.mean(dim=0)
    ovr_max = one_vs_rest.amax(dim=0)
    summary = ovr_mean + alpha * ovr_max
    return summary, ovr_mean, ovr_max


def rank_features(scores: torch.Tensor) -> torch.Tensor:
    return torch.argsort(scores, descending=True)


def standardized_columns(features: torch.Tensor) -> torch.Tensor:
    matrix = features.to(torch.float32)
    mean = matrix.mean(dim=0, keepdim=True)
    std = matrix.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (matrix - mean) / std


def topk_with_correlation_pruning(
    features: torch.Tensor,
    ranked_indices: torch.Tensor,
    top_k: int,
    max_abs_correlation: float,
) -> torch.Tensor:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if max_abs_correlation >= 1.0:
        return ranked_indices[:top_k]

    standardized = standardized_columns(features)
    selected: list[int] = []
    selected_matrix: torch.Tensor | None = None

    for index in ranked_indices.tolist():
        candidate = standardized[:, index]
        if selected_matrix is None:
            selected.append(index)
            selected_matrix = candidate.unsqueeze(1)
        else:
            correlations = torch.abs((selected_matrix * candidate.unsqueeze(1)).mean(dim=0))
            if float(correlations.max().item()) <= max_abs_correlation:
                selected.append(index)
                selected_matrix = torch.cat((selected_matrix, candidate.unsqueeze(1)), dim=1)
        if len(selected) >= top_k:
            break

    return torch.tensor(selected, dtype=torch.int64)


def save_selection(
    path: str | Path,
    scores: torch.Tensor,
    ranked_indices: torch.Tensor,
    metadata: dict,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": scores.to(torch.float32),
            "ranked_indices": ranked_indices.to(torch.int64),
            **metadata,
        },
        output_path,
    )


def load_selection(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)
