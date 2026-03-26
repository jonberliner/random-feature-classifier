from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from random_feature_classifier.config import SineConfig
from random_feature_classifier.layerwise_selection import (
    FAMILY_OVERALL,
    FAMILY_OVR,
    FAMILY_PAIRWISE,
    LayerSelection,
    LayerSpec,
    LayerwiseRandomFeatureStack,
    cache_loader_tensors,
    fisher_score_mean_max,
    load_layerwise_selection,
    pairwise_score_mean_max,
    save_layerwise_selection,
    select_candidate_banks,
    select_candidate_indices,
)


def test_mean_max_fisher_prefers_discriminative_channel() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    mean_summary = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.1, 0.0],
            [5.0, 0.9, 0.1],
            [5.0, 1.0, 0.1],
        ],
        dtype=torch.float32,
    )
    max_summary = mean_summary + 0.05
    scores = fisher_score_mean_max(mean_summary, max_summary, labels, num_classes=2)
    assert int(scores.argmax().item()) == 0


def test_selection_round_trip_and_shape_behavior(tmp_path: Path) -> None:
    specs = [
        LayerSpec(num_candidates=8, keep_k=3, stride=1),
        LayerSpec(num_candidates=10, keep_k=4, stride=2),
    ]
    stack = LayerwiseRandomFeatureStack(specs, seed=3, mode="random_projection", sine=SineConfig())
    images = torch.randn(6, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64)
    loader = torch.utils.data.DataLoader(TensorDataset(images, labels), batch_size=3, shuffle=False)
    cached_inputs, cached_labels = cache_loader_tensors(loader)
    assert torch.equal(cached_labels, labels)

    selected_indices = []
    cached_current_inputs = cached_inputs
    cached_pooled_parts = []
    for layer_index in range(len(specs)):
        selection, _ = stack.select_layer(
            loader,
            selected_indices=selected_indices,
            layer_index=layer_index,
            num_classes=2,
            device=torch.device("cpu"),
            max_abs_correlation=0.99,
        )
        cached_selection, _ = stack.select_layer_from_cached_inputs(
            cached_current_inputs,
            cached_labels,
            layer_index=layer_index,
            num_classes=2,
            batch_size=3,
            device=torch.device("cpu"),
            max_abs_correlation=0.99,
        )
        assert selection.propagated_indices.shape == (specs[layer_index].keep_k,)
        assert selection.readout_indices.numel() == 0
        assert torch.equal(selection.propagated_indices, cached_selection.propagated_indices)
        assert torch.equal(selection.readout_indices, cached_selection.readout_indices)
        selected_indices.append(cached_selection.propagated_indices)
        cached_current_inputs, pooled_part = stack.advance_cached_inputs(
            cached_current_inputs,
            layer_index=layer_index,
            propagated_indices=cached_selection.propagated_indices,
            readout_indices=cached_selection.readout_indices,
            batch_size=3,
            device=torch.device("cpu"),
            cache_dtype=torch.float32,
        )
        cached_pooled_parts.append(pooled_part)

    layer_selections = [
        LayerSelection(propagated_indices=indices, readout_indices=torch.zeros((0,), dtype=torch.int64))
        for indices in selected_indices
    ]
    features, feature_labels = stack.extract_selected_features(loader, layer_selections, device=torch.device("cpu"))
    cached_features = stack.extract_selected_features_from_cached_inputs(
        cached_inputs,
        layer_selections,
        batch_size=3,
        device=torch.device("cpu"),
        cache_dtype=torch.float32,
    )
    assert features.shape == (6, 2 * (3 + 4))
    assert torch.equal(feature_labels, labels)
    assert torch.allclose(cached_features, features)
    assert torch.allclose(torch.cat(cached_pooled_parts, dim=1), features)

    path = tmp_path / "layerwise_selection.pt"
    save_layerwise_selection(str(path), specs, layer_selections, metadata={"seed": 3})
    payload = load_layerwise_selection(str(path))
    assert payload["seed"] == 3
    assert payload["layer_specs"] == specs
    assert len(payload["layer_selections"]) == 2
    assert torch.equal(payload["selected_indices"][0], layer_selections[0].propagated_indices)


def test_correlation_pruning_limits_redundant_channels() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    mean_summary = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 10.0],
            [5.0, 5.0, 10.0],
        ],
        dtype=torch.float32,
    )
    max_summary = mean_summary.clone()
    kept = select_candidate_indices(
        mean_summary,
        max_summary,
        labels,
        num_classes=2,
        keep_k=2,
        max_abs_correlation=0.9,
    )
    assert set(kept.tolist()) == {0, 2}


def test_hybrid_selection_keeps_broad_and_pairwise_specialists() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    mean_summary = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 2.0],
            [3.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    max_summary = mean_summary.clone()

    multiclass_scores = fisher_score_mean_max(mean_summary, max_summary, labels, num_classes=4)
    pairwise_scores = pairwise_score_mean_max(mean_summary, max_summary, labels, num_classes=4, alpha=0.5)
    multiclass_top = int(multiclass_scores.argmax().item())
    pairwise_ranked = torch.argsort(pairwise_scores, descending=True)
    pairwise_specialist = next(index for index in pairwise_ranked.tolist() if index != multiclass_top)
    kept = select_candidate_indices(
        mean_summary,
        max_summary,
        labels,
        num_classes=4,
        keep_k=2,
        max_abs_correlation=0.99,
        strategy="hybrid_pairwise",
        pairwise_alpha=0.5,
        multiclass_keep_fraction=0.5,
    )
    assert multiclass_top in kept.tolist()
    assert pairwise_specialist in kept.tolist()


def test_global_plus_pairwise_per_pair_adds_extra_bank() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    mean_summary = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.5, 0.0, 0.0],
            [1.0, 2.5, 0.0, 0.0],
            [2.0, 0.0, 1.5, 0.0],
            [2.0, 0.0, 1.5, 0.0],
            [3.0, 0.0, 0.0, 2.0],
            [3.0, 0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    max_summary = mean_summary.clone()

    global_scores = fisher_score_mean_max(mean_summary, max_summary, labels, num_classes=4)
    global_top = int(global_scores.argmax().item())
    kept = select_candidate_indices(
        mean_summary,
        max_summary,
        labels,
        num_classes=4,
        keep_k=1,
        pairwise_extra_keep_k=2,
        max_abs_correlation=0.99,
        strategy="global_plus_pairwise_per_pair",
    )
    assert kept.shape == (3,)
    assert kept[0].item() == global_top
    assert len(set(kept.tolist())) == 3


def test_split_bank_selection_propagates_only_global_channels() -> None:
    specs = [
        LayerSpec(num_candidates=8, keep_k=2, stride=1),
        LayerSpec(num_candidates=8, keep_k=2, stride=2),
        LayerSpec(num_candidates=8, keep_k=2, stride=1),
    ]
    pairwise_schedule = [0, 1, 2]
    stack = LayerwiseRandomFeatureStack(specs, seed=5, mode="random_projection", sine=SineConfig())
    images = torch.randn(8, 3, 16, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    cached_current_inputs = images
    layer_selections: list[LayerSelection] = []
    pooled_parts = []

    for layer_index, extra_keep in enumerate(pairwise_schedule):
        selection, _ = stack.select_layer_from_cached_inputs(
            cached_current_inputs,
            labels,
            layer_index=layer_index,
            num_classes=4,
            batch_size=4,
            device=torch.device("cpu"),
            max_abs_correlation=0.99,
            strategy="global_plus_pairwise_per_pair",
            pairwise_extra_keep_k=extra_keep,
        )
        assert selection.propagated_indices.shape == (2,)
        assert selection.readout_indices.shape == (extra_keep,)
        assert set(selection.propagated_indices.tolist()).isdisjoint(selection.readout_indices.tolist())
        cached_current_inputs, pooled = stack.advance_cached_inputs(
            cached_current_inputs,
            layer_index=layer_index,
            propagated_indices=selection.propagated_indices,
            readout_indices=selection.readout_indices,
            batch_size=4,
            device=torch.device("cpu"),
            cache_dtype=torch.float32,
        )
        assert cached_current_inputs.shape[1] == specs[layer_index].keep_k
        assert pooled.shape[1] == 2 * (specs[layer_index].keep_k + extra_keep)
        layer_selections.append(selection)
        pooled_parts.append(pooled)

    features = stack.extract_selected_features_from_cached_inputs(
        images,
        layer_selections,
        batch_size=4,
        device=torch.device("cpu"),
        cache_dtype=torch.float32,
    )
    expected_feature_dim = sum(2 * (spec.keep_k + extra_keep) for spec, extra_keep in zip(specs, pairwise_schedule, strict=True))
    assert features.shape == (8, expected_feature_dim)
    assert torch.allclose(features, torch.cat(pooled_parts, dim=1))


def test_pairwise_refinement_selection_tracks_pair_specific_indices() -> None:
    specs = [
        LayerSpec(num_candidates=8, keep_k=2, stride=1),
        LayerSpec(num_candidates=8, keep_k=2, stride=2),
    ]
    stack = LayerwiseRandomFeatureStack(specs, seed=11, mode="random_projection", sine=SineConfig())
    images = torch.randn(8, 3, 16, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)

    first_selection, _ = stack.select_layer_from_cached_inputs(
        images,
        labels,
        layer_index=0,
        num_classes=4,
        batch_size=4,
        device=torch.device("cpu"),
        max_abs_correlation=0.99,
        strategy="global_pairwise_refinement",
        pairwise_extra_keep_k=2,
    )
    assert first_selection.propagated_indices.shape == (2,)
    assert first_selection.readout_indices.numel() == 0
    assert first_selection.pairwise_indices is not None
    assert first_selection.pairwise_indices.shape == (6, 2)
    for pair_indices in first_selection.pairwise_indices:
        assert set(pair_indices.tolist()).isdisjoint(first_selection.propagated_indices.tolist())

    next_inputs, _ = stack.advance_cached_inputs(
        images,
        layer_index=0,
        propagated_indices=first_selection.propagated_indices,
        readout_indices=None,
        batch_size=4,
        device=torch.device("cpu"),
        cache_dtype=torch.float32,
    )
    second_selection, _ = stack.select_layer_from_cached_inputs(
        next_inputs,
        labels,
        layer_index=1,
        num_classes=4,
        batch_size=4,
        device=torch.device("cpu"),
        max_abs_correlation=0.99,
        strategy="global_pairwise_refinement",
        pairwise_extra_keep_k=1,
    )
    global_features, pairwise_features = stack.extract_global_and_pairwise_features_from_cached_inputs(
        images,
        [first_selection, second_selection],
        batch_size=4,
        device=torch.device("cpu"),
        cache_dtype=torch.float32,
    )
    assert global_features.shape == (8, 2 * (2 + 2))
    assert pairwise_features.shape == (8, 6, 2 * (2 + 1))


def test_multisignal_shortlist_selection_tracks_feature_families() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    mean_summary = torch.tensor(
        [
            [2.5, 0.0, 0.0, 1.2],
            [2.6, 0.0, 0.0, 1.1],
            [0.0, 2.5, 0.0, 1.0],
            [0.0, 2.6, 0.0, 1.1],
            [0.0, 0.0, 2.5, 1.0],
            [0.0, 0.0, 2.6, 1.1],
        ],
        dtype=torch.float32,
    )
    max_summary = mean_summary.clone()
    selection = select_candidate_banks(
        mean_summary,
        max_summary,
        labels,
        num_classes=3,
        keep_k=3,
        max_abs_correlation=1.0,
        strategy="multiclass_pairwise_ovr_shortlist",
        multiclass_keep_fraction=0.34,
        pairwise_keep_fraction=0.33,
        ovr_keep_fraction=0.33,
        shortlist_multiplier=2.0,
    )
    assert selection.propagated_indices.shape == (3,)
    assert selection.family_masks is not None
    assert selection.family_masks.shape == (3,)
    masks = selection.family_masks.tolist()
    assert any(mask & FAMILY_OVERALL for mask in masks)
    assert any(mask & FAMILY_PAIRWISE for mask in masks)
    assert any(mask & FAMILY_OVR for mask in masks)
