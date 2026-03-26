from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .config import SineConfig
from .layers import choose_group_count
from .selection import (
    fisher_scores,
    one_vs_rest_fisher_scores,
    one_vs_rest_fisher_summary,
    pairwise_fisher_scores,
    pairwise_fisher_summary,
    rank_features,
    standardized_columns,
    topk_with_correlation_pruning,
)
from .prng import sign_tensor, uniform_tensor


@dataclass(frozen=True)
class LayerSpec:
    num_candidates: int
    keep_k: int
    stride: int


@dataclass(frozen=True)
class LayerSelection:
    propagated_indices: torch.Tensor
    readout_indices: torch.Tensor
    pairwise_indices: torch.Tensor | None = None
    family_masks: torch.Tensor | None = None

    def all_indices(self) -> torch.Tensor:
        if self.readout_indices.numel() == 0:
            return self.propagated_indices
        return torch.cat((self.propagated_indices, self.readout_indices), dim=0)


def save_layerwise_selection(
    path: str,
    layer_specs: list[LayerSpec],
    layer_selections: list[LayerSelection],
    metadata: dict | None = None,
) -> None:
    payload = {
        "layer_specs": [spec.__dict__ for spec in layer_specs],
        "layer_selections": [
            {
                "propagated_indices": selection.propagated_indices.to(torch.int64).cpu(),
                "readout_indices": selection.readout_indices.to(torch.int64).cpu(),
                "pairwise_indices": None
                if selection.pairwise_indices is None
                else selection.pairwise_indices.to(torch.int64).cpu(),
                "family_masks": None
                if selection.family_masks is None
                else selection.family_masks.to(torch.int64).cpu(),
            }
            for selection in layer_selections
        ],
        "selected_indices": [selection.propagated_indices.to(torch.int64).cpu() for selection in layer_selections],
    }
    if metadata is not None:
        payload.update(metadata)
    torch.save(payload, path)


def load_layerwise_selection(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["layer_specs"] = [LayerSpec(**spec) for spec in payload["layer_specs"]]
    if "layer_selections" in payload:
        payload["layer_selections"] = [
            LayerSelection(
                propagated_indices=selection["propagated_indices"].to(torch.int64),
                readout_indices=selection["readout_indices"].to(torch.int64),
                pairwise_indices=None
                if selection.get("pairwise_indices") is None
                else selection["pairwise_indices"].to(torch.int64),
                family_masks=None
                if selection.get("family_masks") is None
                else selection["family_masks"].to(torch.int64),
            )
            for selection in payload["layer_selections"]
        ]
    else:
        payload["layer_selections"] = [
            LayerSelection(
                propagated_indices=indices.to(torch.int64),
                readout_indices=torch.zeros((0,), dtype=torch.int64),
                pairwise_indices=None,
                family_masks=None,
            )
            for indices in payload["selected_indices"]
        ]
    payload["selected_indices"] = [selection.propagated_indices for selection in payload["layer_selections"]]
    return payload


def candidate_seed_offset(layer_index: int) -> int:
    return 10000 + layer_index * 100


def random_candidate_tensors(
    in_channels: int,
    num_candidates: int,
    kernel_size: int,
    seed: int,
    seed_offset: int,
    mode: str,
    sine: SineConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (num_candidates, in_channels, kernel_size, kernel_size)
    if mode == "strict_ones":
        weight = torch.ones(shape, dtype=torch.float32)
    elif mode == "random_projection":
        weight = sign_tensor(shape, seed, seed_offset, 0)
    else:
        raise ValueError(f"Unknown candidate mode: {mode}")

    a = uniform_tensor((num_candidates,), sine.a_min, sine.a_max, seed, seed_offset, 1).view(1, num_candidates, 1, 1)
    b = uniform_tensor((num_candidates,), sine.b_min, sine.b_max, seed, seed_offset, 2).view(1, num_candidates, 1, 1)
    c = uniform_tensor((num_candidates,), sine.c_min, sine.c_max, seed, seed_offset, 3).view(1, num_candidates, 1, 1)
    return weight, a, b, c


def apply_candidate_layer(
    x: torch.Tensor,
    layer_index: int,
    num_candidates: int,
    stride: int,
    seed: int,
    mode: str,
    sine: SineConfig,
) -> torch.Tensor:
    seed_offset = candidate_seed_offset(layer_index)
    weight, a, b, c = random_candidate_tensors(
        in_channels=x.shape[1],
        num_candidates=num_candidates,
        kernel_size=3,
        seed=seed,
        seed_offset=seed_offset,
        mode=mode,
        sine=sine,
    )
    weight = weight.to(device=x.device, dtype=x.dtype)
    a = a.to(device=x.device, dtype=x.dtype)
    b = b.to(device=x.device, dtype=x.dtype)
    c = c.to(device=x.device, dtype=x.dtype)
    conv = F.conv2d(x, weight, stride=stride, padding=1)
    norm = F.group_norm(conv, num_groups=choose_group_count(num_candidates))
    return c * torch.sin(a * norm + b)


def pooled_summaries(activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return activations.mean(dim=(2, 3)), activations.amax(dim=(2, 3))


def summary_matrix(mean_summary: torch.Tensor, max_summary: torch.Tensor) -> torch.Tensor:
    return torch.cat((mean_summary, max_summary), dim=0)


def fisher_score_mean_max(
    mean_summary: torch.Tensor,
    max_summary: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    return fisher_scores(mean_summary, labels, num_classes) + fisher_scores(max_summary, labels, num_classes)


def pairwise_score_mean_max(
    mean_summary: torch.Tensor,
    max_summary: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float,
) -> torch.Tensor:
    mean_pairwise, _, _ = pairwise_fisher_summary(mean_summary, labels, num_classes, alpha)
    max_pairwise, _, _ = pairwise_fisher_summary(max_summary, labels, num_classes, alpha)
    return mean_pairwise + max_pairwise


def pairwise_score_matrix_mean_max(
    mean_summary: torch.Tensor,
    max_summary: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    return pairwise_fisher_scores(mean_summary, labels, num_classes) + pairwise_fisher_scores(max_summary, labels, num_classes)


def one_vs_rest_score_mean_max(
    mean_summary: torch.Tensor,
    max_summary: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float,
) -> torch.Tensor:
    mean_ovr, _, _ = one_vs_rest_fisher_summary(mean_summary, labels, num_classes, alpha)
    max_ovr, _, _ = one_vs_rest_fisher_summary(max_summary, labels, num_classes, alpha)
    return mean_ovr + max_ovr


def one_vs_rest_score_matrix_mean_max(
    mean_summary: torch.Tensor,
    max_summary: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    return one_vs_rest_fisher_scores(mean_summary, labels, num_classes) + one_vs_rest_fisher_scores(max_summary, labels, num_classes)


FAMILY_OVERALL = 1
FAMILY_PAIRWISE = 2
FAMILY_OVR = 4


def expand_feature_family_masks(layer_selections: list[LayerSelection]) -> torch.Tensor:
    expanded = []
    for selection in layer_selections:
        selected_width = int(selection.all_indices().numel())
        if selected_width == 0:
            continue
        if selection.family_masks is None:
            family_masks = torch.full((selected_width,), FAMILY_OVERALL, dtype=torch.int64)
        else:
            family_masks = selection.family_masks.to(torch.int64)
        expanded.append(family_masks)
        expanded.append(family_masks.clone())
    if not expanded:
        return torch.zeros((0,), dtype=torch.int64)
    return torch.cat(expanded, dim=0)


def normalize_score_vector(scores: torch.Tensor) -> torch.Tensor:
    scores_float = scores.to(torch.float32)
    scale = scores_float.abs().amax().clamp_min(1e-6)
    return scores_float / scale

def correlation_pruned_ranking(
    standardized: torch.Tensor,
    ranked_indices: torch.Tensor,
    target_count: int,
    max_abs_correlation: float,
    selected: list[int] | None = None,
) -> list[int]:
    if target_count <= 0:
        return [] if selected is None else selected

    chosen = [] if selected is None else list(selected)
    chosen_set = set(chosen)
    selected_matrix = standardized[:, chosen] if chosen else None

    for index in ranked_indices.tolist():
        if index in chosen_set:
            continue
        candidate = standardized[:, index]
        if selected_matrix is None:
            chosen.append(index)
            chosen_set.add(index)
            selected_matrix = candidate.unsqueeze(1)
        else:
            correlations = torch.abs((selected_matrix * candidate.unsqueeze(1)).mean(dim=0))
            if float(correlations.max().item()) <= max_abs_correlation:
                chosen.append(index)
                chosen_set.add(index)
                selected_matrix = torch.cat((selected_matrix, candidate.unsqueeze(1)), dim=1)
        if len(chosen) >= target_count:
            break

    return chosen


def select_pairwise_per_pair_indices(
    pairwise_scores: torch.Tensor,
    target_count: int,
    excluded: list[int] | None = None,
) -> list[int]:
    if target_count <= 0:
        return []
    if pairwise_scores.shape[0] == 0:
        return []

    num_pairs, num_features = pairwise_scores.shape
    excluded_set = set([] if excluded is None else excluded)
    selected: list[int] = []
    selected_set = set(excluded_set)

    pair_rankings = torch.argsort(pairwise_scores, dim=1, descending=True)
    pair_strength = pairwise_scores.amax(dim=1)
    pair_order = torch.argsort(pair_strength, descending=True).tolist()
    base_quota = target_count // num_pairs
    remainder = target_count % num_pairs

    for pair_order_index, pair_index in enumerate(pair_order):
        quota = base_quota + (1 if pair_order_index < remainder else 0)
        if quota <= 0:
            continue
        added = 0
        for feature_index in pair_rankings[pair_index].tolist():
            if feature_index in selected_set:
                continue
            selected.append(feature_index)
            selected_set.add(feature_index)
            added += 1
            if added >= quota or len(selected) >= target_count:
                break
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        flat_rank = torch.argsort(pairwise_scores.reshape(-1), descending=True)
        for flat_index in flat_rank.tolist():
            feature_index = flat_index % num_features
            if feature_index in selected_set:
                continue
            selected.append(feature_index)
            selected_set.add(feature_index)
            if len(selected) >= target_count:
                break

    return selected[:target_count]


def select_groupwise_indices(
    group_scores: torch.Tensor,
    target_count: int,
    excluded: list[int] | None = None,
) -> list[int]:
    if target_count <= 0 or group_scores.shape[0] == 0:
        return []

    num_groups, num_features = group_scores.shape
    excluded_set = set([] if excluded is None else excluded)
    selected: list[int] = []
    selected_set = set(excluded_set)

    group_rankings = torch.argsort(group_scores, dim=1, descending=True)
    group_strength = group_scores.amax(dim=1)
    group_order = torch.argsort(group_strength, descending=True).tolist()
    base_quota = target_count // num_groups
    remainder = target_count % num_groups

    for group_order_index, group_index in enumerate(group_order):
        quota = base_quota + (1 if group_order_index < remainder else 0)
        if quota <= 0:
            continue
        added = 0
        for feature_index in group_rankings[group_index].tolist():
            if feature_index in selected_set:
                continue
            selected.append(feature_index)
            selected_set.add(feature_index)
            added += 1
            if added >= quota or len(selected) >= target_count:
                break
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        flat_rank = torch.argsort(group_scores.reshape(-1), descending=True)
        for flat_index in flat_rank.tolist():
            feature_index = flat_index % num_features
            if feature_index in selected_set:
                continue
            selected.append(feature_index)
            selected_set.add(feature_index)
            if len(selected) >= target_count:
                break

    return selected[:target_count]


def select_pairwise_per_pair_bank(
    features: torch.Tensor,
    pairwise_scores: torch.Tensor,
    keep_per_pair: int,
    max_abs_correlation: float,
    excluded_global: list[int],
) -> torch.Tensor:
    num_pairs = pairwise_scores.shape[0]
    if keep_per_pair <= 0 or num_pairs == 0:
        return torch.zeros((num_pairs, 0), dtype=torch.int64)

    standardized = standardized_columns(features)
    banks = []
    for pair_index in range(num_pairs):
        ranked = torch.argsort(pairwise_scores[pair_index], descending=True)
        selected = correlation_pruned_ranking(
            standardized,
            ranked,
            target_count=len(excluded_global) + keep_per_pair,
            max_abs_correlation=max_abs_correlation,
            selected=excluded_global,
        )
        pair_selected = selected[len(excluded_global) :]
        selected_set = set(excluded_global) | set(pair_selected)
        if len(pair_selected) < keep_per_pair:
            for feature_index in ranked.tolist():
                if feature_index in selected_set:
                    continue
                pair_selected.append(feature_index)
                selected_set.add(feature_index)
                if len(pair_selected) >= keep_per_pair:
                    break
        banks.append(torch.tensor(pair_selected[:keep_per_pair], dtype=torch.int64))

    return torch.stack(banks, dim=0)


def select_candidate_indices(
    mean_summary: torch.Tensor,
    max_summary: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    keep_k: int,
    max_abs_correlation: float,
    strategy: str = "multiclass",
    pairwise_alpha: float = 0.5,
    multiclass_keep_fraction: float = 0.5,
    pairwise_extra_keep_k: int = 0,
    pairwise_keep_fraction: float = 0.25,
    ovr_keep_fraction: float = 0.25,
    shortlist_multiplier: float = 2.0,
) -> torch.Tensor:
    selection = select_candidate_banks(
        mean_summary,
        max_summary,
        labels,
        num_classes=num_classes,
        keep_k=keep_k,
        max_abs_correlation=max_abs_correlation,
        strategy=strategy,
        pairwise_alpha=pairwise_alpha,
        multiclass_keep_fraction=multiclass_keep_fraction,
        pairwise_extra_keep_k=pairwise_extra_keep_k,
        pairwise_keep_fraction=pairwise_keep_fraction,
        ovr_keep_fraction=ovr_keep_fraction,
        shortlist_multiplier=shortlist_multiplier,
    )
    return selection.all_indices()


def select_candidate_banks(
    mean_summary: torch.Tensor,
    max_summary: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    keep_k: int,
    max_abs_correlation: float,
    strategy: str = "multiclass",
    pairwise_alpha: float = 0.5,
    multiclass_keep_fraction: float = 0.5,
    pairwise_extra_keep_k: int = 0,
    pairwise_keep_fraction: float = 0.25,
    ovr_keep_fraction: float = 0.25,
    shortlist_multiplier: float = 2.0,
) -> LayerSelection:
    scores = fisher_score_mean_max(mean_summary, max_summary, labels, num_classes)
    ranked = rank_features(scores)
    features = summary_matrix(mean_summary, max_summary)

    if strategy == "multiclass":
        pruned = topk_with_correlation_pruning(
            features,
            ranked,
            top_k=keep_k,
            max_abs_correlation=max_abs_correlation,
        )
        selected = pruned.tolist()
        selected_set = set(selected)
        for index in ranked.tolist():
            if len(selected) >= keep_k:
                break
            if index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
        return LayerSelection(
            propagated_indices=torch.tensor(selected[:keep_k], dtype=torch.int64),
            readout_indices=torch.zeros((0,), dtype=torch.int64),
            family_masks=torch.full((keep_k,), FAMILY_OVERALL, dtype=torch.int64),
        )

    if strategy == "global_plus_pairwise_per_pair":
        if pairwise_extra_keep_k < 0:
            raise ValueError("pairwise_extra_keep_k must be non-negative for global_plus_pairwise_per_pair.")
        global_pruned = topk_with_correlation_pruning(
            features,
            ranked,
            top_k=keep_k,
            max_abs_correlation=max_abs_correlation,
        )
        global_list = global_pruned.tolist()
        global_set = set(global_list)
        for index in ranked.tolist():
            if len(global_list) >= keep_k:
                break
            if index in global_set:
                continue
            global_list.append(index)
            global_set.add(index)

        if pairwise_extra_keep_k == 0:
            return LayerSelection(
                propagated_indices=torch.tensor(global_list[:keep_k], dtype=torch.int64),
                readout_indices=torch.zeros((0,), dtype=torch.int64),
                family_masks=torch.full((keep_k,), FAMILY_OVERALL, dtype=torch.int64),
            )

        pairwise_scores = pairwise_score_matrix_mean_max(mean_summary, max_summary, labels, num_classes)
        pairwise_priority = select_pairwise_per_pair_indices(
            pairwise_scores,
            target_count=max(pairwise_extra_keep_k * 2, pairwise_extra_keep_k),
            excluded=global_list,
        )
        standardized = standardized_columns(features)
        combined = correlation_pruned_ranking(
            standardized,
            torch.tensor(pairwise_priority, dtype=torch.int64),
            target_count=keep_k + pairwise_extra_keep_k,
            max_abs_correlation=max_abs_correlation,
            selected=global_list[:keep_k],
        )

        if len(combined) < keep_k + pairwise_extra_keep_k:
            num_features = pairwise_scores.shape[1]
            flat_rank = torch.argsort(pairwise_scores.reshape(-1), descending=True)
            flat_feature_rank = []
            seen = set()
            for flat_index in flat_rank.tolist():
                feature_index = flat_index % num_features
                if feature_index in seen:
                    continue
                flat_feature_rank.append(feature_index)
                seen.add(feature_index)
            combined = correlation_pruned_ranking(
                standardized,
                torch.tensor(flat_feature_rank, dtype=torch.int64),
                target_count=keep_k + pairwise_extra_keep_k,
                max_abs_correlation=max_abs_correlation,
                selected=combined,
            )

        if len(combined) < keep_k + pairwise_extra_keep_k:
            combined_set = set(combined)
            for fallback in (ranked.tolist(), torch.argsort(pairwise_scores.reshape(-1), descending=True).tolist()):
                for value in fallback:
                    feature_index = value if isinstance(value, int) and value < features.shape[1] else value % features.shape[1]
                    if feature_index in combined_set:
                        continue
                    combined.append(feature_index)
                    combined_set.add(feature_index)
                    if len(combined) >= keep_k + pairwise_extra_keep_k:
                        break
                if len(combined) >= keep_k + pairwise_extra_keep_k:
                    break

        propagated = torch.tensor(global_list[:keep_k], dtype=torch.int64)
        readout = torch.tensor(combined[keep_k : keep_k + pairwise_extra_keep_k], dtype=torch.int64)
        family_masks = torch.cat(
            (
                torch.full((propagated.numel(),), FAMILY_OVERALL, dtype=torch.int64),
                torch.full((readout.numel(),), FAMILY_PAIRWISE, dtype=torch.int64),
            ),
            dim=0,
        )
        return LayerSelection(
            propagated_indices=propagated,
            readout_indices=readout,
            pairwise_indices=None,
            family_masks=family_masks,
        )

    if strategy == "global_pairwise_refinement":
        if pairwise_extra_keep_k < 0:
            raise ValueError("pairwise_extra_keep_k must be non-negative for global_pairwise_refinement.")
        global_pruned = topk_with_correlation_pruning(
            features,
            ranked,
            top_k=keep_k,
            max_abs_correlation=max_abs_correlation,
        )
        global_list = global_pruned.tolist()
        global_set = set(global_list)
        for index in ranked.tolist():
            if len(global_list) >= keep_k:
                break
            if index in global_set:
                continue
            global_list.append(index)
            global_set.add(index)

        if pairwise_extra_keep_k == 0:
            return LayerSelection(
                propagated_indices=torch.tensor(global_list[:keep_k], dtype=torch.int64),
                readout_indices=torch.zeros((0,), dtype=torch.int64),
                pairwise_indices=torch.zeros((0, 0), dtype=torch.int64),
                family_masks=torch.full((keep_k,), FAMILY_OVERALL, dtype=torch.int64),
            )

        pairwise_scores = pairwise_score_matrix_mean_max(mean_summary, max_summary, labels, num_classes)
        pairwise_bank = select_pairwise_per_pair_bank(
            features,
            pairwise_scores,
            keep_per_pair=pairwise_extra_keep_k,
            max_abs_correlation=max_abs_correlation,
            excluded_global=global_list[:keep_k],
        )
        return LayerSelection(
            propagated_indices=torch.tensor(global_list[:keep_k], dtype=torch.int64),
            readout_indices=torch.zeros((0,), dtype=torch.int64),
            pairwise_indices=pairwise_bank,
            family_masks=torch.full((keep_k,), FAMILY_OVERALL, dtype=torch.int64),
        )

    if strategy == "multiclass_pairwise_ovr_shortlist":
        if shortlist_multiplier < 1.0:
            raise ValueError("shortlist_multiplier must be at least 1.0.")
        if multiclass_keep_fraction < 0.0 or pairwise_keep_fraction < 0.0 or ovr_keep_fraction < 0.0:
            raise ValueError("Family keep fractions must be non-negative.")

        total_fraction = multiclass_keep_fraction + pairwise_keep_fraction + ovr_keep_fraction
        if total_fraction <= 0.0:
            raise ValueError("At least one family keep fraction must be positive.")

        pairwise_scores = pairwise_score_matrix_mean_max(mean_summary, max_summary, labels, num_classes)
        ovr_scores = one_vs_rest_score_matrix_mean_max(mean_summary, max_summary, labels, num_classes)
        pairwise_best = pairwise_scores.amax(dim=0) if pairwise_scores.shape[0] > 0 else torch.zeros_like(scores)
        ovr_best = ovr_scores.amax(dim=0) if ovr_scores.shape[0] > 0 else torch.zeros_like(scores)

        shortlist_total = max(keep_k, int(round(keep_k * shortlist_multiplier)))
        overall_target = max(1, int(round(shortlist_total * multiclass_keep_fraction / total_fraction)))
        pairwise_target = max(0, int(round(shortlist_total * pairwise_keep_fraction / total_fraction)))
        ovr_target = max(0, shortlist_total - overall_target - pairwise_target)

        overall_shortlist = ranked[: min(overall_target, ranked.numel())].tolist()
        pairwise_shortlist = select_groupwise_indices(pairwise_scores, pairwise_target)
        ovr_shortlist = select_groupwise_indices(ovr_scores, ovr_target)

        combined_candidates: list[int] = []
        combined_set: set[int] = set()
        for candidate_list in (overall_shortlist, pairwise_shortlist, ovr_shortlist):
            for feature_index in candidate_list:
                if feature_index in combined_set:
                    continue
                combined_candidates.append(feature_index)
                combined_set.add(feature_index)

        combined_scores = (
            normalize_score_vector(scores)
            + normalize_score_vector(pairwise_best)
            + normalize_score_vector(ovr_best)
        )
        combined_candidates.sort(key=lambda feature_index: float(combined_scores[feature_index].item()), reverse=True)
        candidate_ranking = torch.tensor(combined_candidates, dtype=torch.int64)
        shortlist_features = features[:, candidate_ranking]
        shortlist_standardized = standardized_columns(shortlist_features)
        local_ranking = torch.arange(candidate_ranking.numel(), dtype=torch.int64)
        selected_local = correlation_pruned_ranking(
            shortlist_standardized,
            local_ranking,
            target_count=keep_k,
            max_abs_correlation=max_abs_correlation,
        )
        selected = candidate_ranking[torch.tensor(selected_local, dtype=torch.int64)].tolist()
        selected_set = set(selected)
        fallback_rankings = (
            candidate_ranking.tolist(),
            torch.argsort(combined_scores, descending=True).tolist(),
            ranked.tolist(),
        )
        for fallback_ranking in fallback_rankings:
            for feature_index in fallback_ranking:
                if feature_index in selected_set:
                    continue
                selected.append(feature_index)
                selected_set.add(feature_index)
                if len(selected) >= keep_k:
                    break
            if len(selected) >= keep_k:
                break

        overall_set = set(overall_shortlist)
        pairwise_set = set(pairwise_shortlist)
        ovr_set = set(ovr_shortlist)
        family_masks = []
        for feature_index in selected[:keep_k]:
            mask = 0
            if feature_index in overall_set:
                mask |= FAMILY_OVERALL
            if feature_index in pairwise_set:
                mask |= FAMILY_PAIRWISE
            if feature_index in ovr_set:
                mask |= FAMILY_OVR
            if mask == 0:
                family_scores = {
                    FAMILY_OVERALL: float(scores[feature_index].item()),
                    FAMILY_PAIRWISE: float(pairwise_best[feature_index].item()),
                    FAMILY_OVR: float(ovr_best[feature_index].item()),
                }
                mask = max(family_scores, key=family_scores.get)
            family_masks.append(mask)

        return LayerSelection(
            propagated_indices=torch.tensor(selected[:keep_k], dtype=torch.int64),
            readout_indices=torch.zeros((0,), dtype=torch.int64),
            pairwise_indices=None,
            family_masks=torch.tensor(family_masks, dtype=torch.int64),
        )

    if strategy != "hybrid_pairwise":
        raise ValueError(f"Unknown selection strategy: {strategy}")

    pairwise_scores = pairwise_score_mean_max(
        mean_summary,
        max_summary,
        labels,
        num_classes=num_classes,
        alpha=pairwise_alpha,
    )
    pairwise_ranked = rank_features(pairwise_scores)
    multiclass_target = min(keep_k - 1, max(1, int(round(keep_k * multiclass_keep_fraction))))
    pairwise_target = max(1, keep_k - multiclass_target)
    standardized = standardized_columns(features)

    selected = correlation_pruned_ranking(
        standardized,
        ranked,
        target_count=multiclass_target,
        max_abs_correlation=max_abs_correlation,
    )
    selected = correlation_pruned_ranking(
        standardized,
        pairwise_ranked,
        target_count=multiclass_target + pairwise_target,
        max_abs_correlation=max_abs_correlation,
        selected=selected,
    )

    for fallback_ranking in (pairwise_ranked, ranked):
        selected = correlation_pruned_ranking(
            standardized,
            fallback_ranking,
            target_count=keep_k,
            max_abs_correlation=max_abs_correlation,
            selected=selected,
        )
        if len(selected) >= keep_k:
            break

    if len(selected) < keep_k:
        selected_set = set(selected)
        for fallback_ranking in (pairwise_ranked, ranked):
            for index in fallback_ranking.tolist():
                if index in selected_set:
                    continue
                selected.append(index)
                selected_set.add(index)
                if len(selected) == keep_k:
                    break
            if len(selected) == keep_k:
                break

    return LayerSelection(
        propagated_indices=torch.tensor(selected[:keep_k], dtype=torch.int64),
        readout_indices=torch.zeros((0,), dtype=torch.int64),
        pairwise_indices=None,
        family_masks=torch.full((keep_k,), FAMILY_OVERALL | FAMILY_PAIRWISE, dtype=torch.int64),
    )


def cache_loader_tensors(loader) -> tuple[torch.Tensor, torch.Tensor]:
    input_batches = []
    label_batches = []
    for inputs, labels in loader:
        input_batches.append(inputs.to("cpu"))
        label_batches.append(labels.to(torch.int64))
    return torch.cat(input_batches, dim=0), torch.cat(label_batches, dim=0)


def batch_slice_range(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield slice(start, min(length, start + batch_size))


class LayerwiseRandomFeatureStack:
    def __init__(
        self,
        layer_specs: list[LayerSpec],
        seed: int,
        mode: str,
        sine: SineConfig,
    ) -> None:
        self.layer_specs = layer_specs
        self.seed = seed
        self.mode = mode
        self.sine = sine

    def candidate_activations(self, images: torch.Tensor, propagated_indices: list[torch.Tensor], layer_index: int) -> torch.Tensor:
        current = images
        for current_layer in range(layer_index + 1):
            spec = self.layer_specs[current_layer]
            activations = apply_candidate_layer(
                current,
                layer_index=current_layer,
                num_candidates=spec.num_candidates,
                stride=spec.stride,
                seed=self.seed,
                mode=self.mode,
                sine=self.sine,
            )
            if current_layer == layer_index:
                return activations
            kept = propagated_indices[current_layer].to(device=activations.device)
            current = activations.index_select(1, kept)
        raise RuntimeError("Layer traversal failed.")

    def select_layer(
        self,
        loader,
        selected_indices: list[torch.Tensor],
        layer_index: int,
        num_classes: int,
        device: torch.device,
        max_abs_correlation: float,
        strategy: str = "multiclass",
        pairwise_alpha: float = 0.5,
        multiclass_keep_fraction: float = 0.5,
        pairwise_extra_keep_k: int = 0,
        pairwise_keep_fraction: float = 0.25,
        ovr_keep_fraction: float = 0.25,
        shortlist_multiplier: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_batches = []
        max_batches = []
        label_batches = []

        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                activations = self.candidate_activations(images, selected_indices, layer_index)
                mean_summary, max_summary = pooled_summaries(activations)
                mean_batches.append(mean_summary.to("cpu", dtype=torch.float32))
                max_batches.append(max_summary.to("cpu", dtype=torch.float32))
                label_batches.append(labels.to(torch.int64))

        mean_summary = torch.cat(mean_batches, dim=0)
        max_summary = torch.cat(max_batches, dim=0)
        labels = torch.cat(label_batches, dim=0)
        selection = select_candidate_banks(
            mean_summary,
            max_summary,
            labels,
            num_classes=num_classes,
            keep_k=self.layer_specs[layer_index].keep_k,
            max_abs_correlation=max_abs_correlation,
            strategy=strategy,
            pairwise_alpha=pairwise_alpha,
            multiclass_keep_fraction=multiclass_keep_fraction,
            pairwise_extra_keep_k=pairwise_extra_keep_k,
            pairwise_keep_fraction=pairwise_keep_fraction,
            ovr_keep_fraction=ovr_keep_fraction,
            shortlist_multiplier=shortlist_multiplier,
        )
        return selection, fisher_score_mean_max(mean_summary, max_summary, labels, num_classes)

    def select_layer_from_cached_inputs(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        layer_index: int,
        num_classes: int,
        batch_size: int,
        device: torch.device,
        max_abs_correlation: float,
        strategy: str = "multiclass",
        pairwise_alpha: float = 0.5,
        multiclass_keep_fraction: float = 0.5,
        pairwise_extra_keep_k: int = 0,
        pairwise_keep_fraction: float = 0.25,
        ovr_keep_fraction: float = 0.25,
        shortlist_multiplier: float = 2.0,
    ) -> tuple[LayerSelection, torch.Tensor]:
        mean_batches = []
        max_batches = []

        with torch.inference_mode():
            for batch_slice in batch_slice_range(inputs.shape[0], batch_size):
                batch_inputs = inputs[batch_slice].to(device, non_blocking=True)
                if device.type == "cpu" and batch_inputs.dtype == torch.float16:
                    batch_inputs = batch_inputs.to(torch.float32)
                activations = apply_candidate_layer(
                    batch_inputs,
                    layer_index=layer_index,
                    num_candidates=self.layer_specs[layer_index].num_candidates,
                    stride=self.layer_specs[layer_index].stride,
                    seed=self.seed,
                    mode=self.mode,
                    sine=self.sine,
                )
                mean_summary, max_summary = pooled_summaries(activations)
                mean_batches.append(mean_summary.to("cpu", dtype=torch.float32))
                max_batches.append(max_summary.to("cpu", dtype=torch.float32))

        mean_summary = torch.cat(mean_batches, dim=0)
        max_summary = torch.cat(max_batches, dim=0)
        selection = select_candidate_banks(
            mean_summary,
            max_summary,
            labels,
            num_classes=num_classes,
            keep_k=self.layer_specs[layer_index].keep_k,
            max_abs_correlation=max_abs_correlation,
            strategy=strategy,
            pairwise_alpha=pairwise_alpha,
            multiclass_keep_fraction=multiclass_keep_fraction,
            pairwise_extra_keep_k=pairwise_extra_keep_k,
            pairwise_keep_fraction=pairwise_keep_fraction,
            ovr_keep_fraction=ovr_keep_fraction,
            shortlist_multiplier=shortlist_multiplier,
        )
        return selection, fisher_score_mean_max(mean_summary, max_summary, labels, num_classes)

    def advance_cached_inputs(
        self,
        inputs: torch.Tensor,
        layer_index: int,
        propagated_indices: torch.Tensor,
        readout_indices: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        cache_dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_shape: tuple[int, int, int, int] | None = None
        cached_inputs: torch.Tensor | None = None
        readout_dim = int(readout_indices.numel()) if readout_indices is not None else 0
        pooled_features = torch.empty((inputs.shape[0], 2 * (propagated_indices.numel() + readout_dim)), dtype=torch.float32)

        with torch.inference_mode():
            propagated_device = propagated_indices.to(device)
            readout_device = None if readout_indices is None else readout_indices.to(device)
            cursor = 0
            for batch_slice in batch_slice_range(inputs.shape[0], batch_size):
                batch_inputs = inputs[batch_slice].to(device, non_blocking=True)
                if device.type == "cpu" and batch_inputs.dtype == torch.float16:
                    batch_inputs = batch_inputs.to(torch.float32)
                activations = apply_candidate_layer(
                    batch_inputs,
                    layer_index=layer_index,
                    num_candidates=self.layer_specs[layer_index].num_candidates,
                    stride=self.layer_specs[layer_index].stride,
                    seed=self.seed,
                    mode=self.mode,
                    sine=self.sine,
                )
                propagated = activations.index_select(1, propagated_device)
                propagated_mean, propagated_max = pooled_summaries(propagated)
                pooled_parts = [propagated_mean, propagated_max]
                batch_size_actual = propagated.shape[0]

                if readout_device is not None and readout_indices.numel() > 0:
                    readout = activations.index_select(1, readout_device)
                    readout_mean, readout_max = pooled_summaries(readout)
                    pooled_parts.extend((readout_mean, readout_max))

                if cached_inputs is None:
                    output_shape = (
                        inputs.shape[0],
                        propagated.shape[1],
                        propagated.shape[2],
                        propagated.shape[3],
                    )
                    cached_inputs = torch.empty(output_shape, dtype=cache_dtype)

                cached_inputs[cursor : cursor + batch_size_actual].copy_(propagated.to("cpu", dtype=cache_dtype))
                pooled_features[cursor : cursor + batch_size_actual].copy_(
                    torch.cat(pooled_parts, dim=1).to("cpu", dtype=torch.float32)
                )
                cursor += batch_size_actual

        if cached_inputs is None or output_shape is None:
            raise RuntimeError("No cached activations were created.")
        return cached_inputs, pooled_features

    def extract_selected_features_from_cached_inputs(
        self,
        inputs: torch.Tensor,
        layer_selections: list[LayerSelection],
        batch_size: int,
        device: torch.device,
        cache_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        current = inputs
        pooled_parts = []
        for layer_index, selection in enumerate(layer_selections):
            current, pooled = self.advance_cached_inputs(
                current,
                layer_index=layer_index,
                propagated_indices=selection.propagated_indices,
                readout_indices=selection.readout_indices,
                batch_size=batch_size,
                device=device,
                cache_dtype=cache_dtype,
            )
            pooled_parts.append(pooled)
        return torch.cat(pooled_parts, dim=1)

    def extract_global_and_pairwise_features_from_cached_inputs(
        self,
        inputs: torch.Tensor,
        layer_selections: list[LayerSelection],
        batch_size: int,
        device: torch.device,
        cache_dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = inputs
        global_parts = []
        pairwise_parts = []

        for layer_index, selection in enumerate(layer_selections):
            next_inputs_shape: tuple[int, int, int, int] | None = None
            next_inputs: torch.Tensor | None = None
            global_layer_features = torch.empty((inputs.shape[0], 2 * selection.propagated_indices.numel()), dtype=torch.float32)
            pairwise_dim = 0 if selection.pairwise_indices is None else int(selection.pairwise_indices.shape[1] * 2)
            pairwise_layer_features = torch.empty(
                (
                    inputs.shape[0],
                    0 if selection.pairwise_indices is None else selection.pairwise_indices.shape[0],
                    pairwise_dim,
                ),
                dtype=torch.float32,
            )

            with torch.inference_mode():
                propagated_device = selection.propagated_indices.to(device)
                pairwise_device = None if selection.pairwise_indices is None else selection.pairwise_indices.to(device)
                cursor = 0
                for batch_slice in batch_slice_range(current.shape[0], batch_size):
                    batch_inputs = current[batch_slice].to(device, non_blocking=True)
                    if device.type == "cpu" and batch_inputs.dtype == torch.float16:
                        batch_inputs = batch_inputs.to(torch.float32)
                    activations = apply_candidate_layer(
                        batch_inputs,
                        layer_index=layer_index,
                        num_candidates=self.layer_specs[layer_index].num_candidates,
                        stride=self.layer_specs[layer_index].stride,
                        seed=self.seed,
                        mode=self.mode,
                        sine=self.sine,
                    )
                    propagated = activations.index_select(1, propagated_device)
                    global_mean, global_max = pooled_summaries(propagated)
                    batch_size_actual = propagated.shape[0]

                    if next_inputs is None:
                        next_inputs_shape = (
                            current.shape[0],
                            propagated.shape[1],
                            propagated.shape[2],
                            propagated.shape[3],
                        )
                        next_inputs = torch.empty(next_inputs_shape, dtype=cache_dtype)

                    next_inputs[cursor : cursor + batch_size_actual].copy_(propagated.to("cpu", dtype=cache_dtype))
                    global_layer_features[cursor : cursor + batch_size_actual].copy_(
                        torch.cat((global_mean, global_max), dim=1).to("cpu", dtype=torch.float32)
                    )

                    if pairwise_device is not None and pairwise_device.numel() > 0:
                        per_pair_features = []
                        for pair_indices in pairwise_device:
                            pair_acts = activations.index_select(1, pair_indices)
                            pair_mean, pair_max = pooled_summaries(pair_acts)
                            per_pair_features.append(torch.cat((pair_mean, pair_max), dim=1))
                        pairwise_batch = torch.stack(per_pair_features, dim=1).to("cpu", dtype=torch.float32)
                        pairwise_layer_features[cursor : cursor + batch_size_actual].copy_(pairwise_batch)

                    cursor += batch_size_actual

            if next_inputs is None or next_inputs_shape is None:
                raise RuntimeError("No propagated inputs were created.")

            current = next_inputs
            global_parts.append(global_layer_features)
            if pairwise_layer_features.shape[1] > 0:
                pairwise_parts.append(pairwise_layer_features)

        global_features = torch.cat(global_parts, dim=1)
        if not pairwise_parts:
            pairwise_features = torch.zeros((inputs.shape[0], 0, 0), dtype=torch.float32)
        else:
            pairwise_features = torch.cat(pairwise_parts, dim=2)
        return global_features, pairwise_features

    def extract_selected_features(
        self,
        loader,
        layer_selections: list[LayerSelection],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_batches = []
        label_batches = []

        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                current = images
                pooled_parts = []
                for layer_index, spec in enumerate(self.layer_specs):
                    activations = apply_candidate_layer(
                        current,
                        layer_index=layer_index,
                        num_candidates=spec.num_candidates,
                        stride=spec.stride,
                        seed=self.seed,
                        mode=self.mode,
                        sine=self.sine,
                    )
                    selection = layer_selections[layer_index]
                    propagated = activations.index_select(1, selection.propagated_indices.to(device=activations.device))
                    current = propagated
                    propagated_mean, propagated_max = pooled_summaries(propagated)
                    pooled_blocks = [propagated_mean, propagated_max]
                    if selection.readout_indices.numel() > 0:
                        readout = activations.index_select(1, selection.readout_indices.to(device=activations.device))
                        readout_mean, readout_max = pooled_summaries(readout)
                        pooled_blocks.extend((readout_mean, readout_max))
                    pooled_parts.append(torch.cat(pooled_blocks, dim=1))

                feature_batches.append(torch.cat(pooled_parts, dim=1).to("cpu", dtype=torch.float32))
                label_batches.append(labels.to(torch.int64))

        return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)
