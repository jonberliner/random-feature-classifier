from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from torch import nn

from .cache import load_matrix_feature_cache
from .config import SineConfig, TrainConfig
from .data import build_feature_loader
from .layerwise_selection import (
    FAMILY_OVERALL,
    FAMILY_OVR,
    FAMILY_PAIRWISE,
    LayerSelection,
    LayerwiseRandomFeatureStack,
    cache_loader_tensors,
    expand_feature_family_masks,
    load_layerwise_selection,
)
from .pairwise import class_pairs
from .selection import fisher_scores, pairwise_fisher_scores, standardized_columns
from .train_head import make_scheduler, standardize_train_test
from .train_layerwise_selected import build_eval_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze selected random features and save diagnostic plots.")
    parser.add_argument("--selection-path", required=True)
    parser.add_argument("--train-cache")
    parser.add_argument("--test-cache")
    parser.add_argument("--output-dir", default="experiment_results/fisher_features_analysis")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--top-correlation-features", type=int, default=128)
    parser.add_argument("--top-scatter-features", type=int, default=512)
    parser.add_argument("--projection-samples-per-split", type=int, default=2000)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--head-batch-size", type=int, default=1024)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-weight-decay", type=float, default=5e-4)
    parser.add_argument("--head-label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    return parser.parse_args()


def output_subdir(base_dir: str, selection_path: str) -> Path:
    selection_name = Path(selection_path).stem
    return Path(base_dir) / selection_name


def feature_block_slices(layer_selections: list[LayerSelection]) -> list[slice]:
    slices = []
    cursor = 0
    for selection in layer_selections:
        block_width = 2 * (int(selection.propagated_indices.numel()) + int(selection.readout_indices.numel()))
        slices.append(slice(cursor, cursor + block_width))
        cursor += block_width
    return slices


def feature_layer_ids(block_slices: list[slice]) -> torch.Tensor:
    layer_ids = []
    for layer_index, feature_slice in enumerate(block_slices):
        layer_ids.extend([layer_index] * (feature_slice.stop - feature_slice.start))
    return torch.tensor(layer_ids, dtype=torch.int64)


def feature_block_slices_from_layer_ids(layer_ids: torch.Tensor, num_layers: int) -> list[slice]:
    slices = []
    cursor = 0
    for layer_index in range(num_layers):
        layer_count = int((layer_ids == layer_index).sum().item())
        slices.append(slice(cursor, cursor + layer_count))
        cursor += layer_count
    return slices


def family_definitions(family_masks: torch.Tensor | None) -> list[tuple[str, int | None]]:
    definitions = [("combined", None)]
    if family_masks is None or family_masks.numel() == 0:
        return definitions
    definitions.extend(
        [
            ("overall", FAMILY_OVERALL),
            ("pairwise", FAMILY_PAIRWISE),
            ("one_vs_rest", FAMILY_OVR),
        ]
    )
    return definitions


def extract_selected_features(selection_payload: dict, batch_size: int, num_workers: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_config = TrainConfig(batch_size=batch_size, num_workers=num_workers)
    train_loader, test_loader = build_eval_loaders(train_config)
    train_inputs, train_labels = cache_loader_tensors(train_loader)
    test_inputs, test_labels = cache_loader_tensors(test_loader)

    selector = LayerwiseRandomFeatureStack(
        layer_specs=selection_payload["layer_specs"],
        seed=int(selection_payload.get("seed", 0)),
        mode=str(selection_payload.get("backbone_kind", "random_projection")),
        sine=SineConfig(),
    )
    train_features = selector.extract_selected_features_from_cached_inputs(
        train_inputs,
        selection_payload["layer_selections"],
        batch_size=batch_size,
        device=device,
    )
    test_features = selector.extract_selected_features_from_cached_inputs(
        test_inputs,
        selection_payload["layer_selections"],
        batch_size=batch_size,
        device=device,
    )
    return train_features, train_labels, test_features, test_labels


def load_or_extract_features(
    selection_payload: dict,
    train_cache: str | None,
    test_cache: str | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_cache is None or test_cache is None:
        return extract_selected_features(selection_payload, batch_size=batch_size, num_workers=num_workers, device=device)

    train_features, train_labels, _ = load_matrix_feature_cache(train_cache)
    test_features, test_labels, _ = load_matrix_feature_cache(test_cache)
    return train_features.to(torch.float32), train_labels, test_features.to(torch.float32), test_labels


def stratified_subset_indices(labels: torch.Tensor, per_class: int) -> torch.Tensor:
    selected = []
    num_classes = int(labels.max().item()) + 1
    for class_index in range(num_classes):
        class_indices = torch.nonzero(labels == class_index, as_tuple=False).flatten()
        selected.append(class_indices[: min(per_class, class_indices.numel())])
    return torch.cat(selected, dim=0)


def save_correlation_heatmap(path: Path, features: torch.Tensor, ranked_indices: torch.Tensor, top_n: int) -> None:
    top_indices = ranked_indices[: min(top_n, ranked_indices.numel())]
    standardized = standardized_columns(features[:, top_indices])
    correlation = torch.abs((standardized.T @ standardized) / standardized.shape[0]).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(correlation, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Absolute Correlation Heatmap")
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("Feature rank")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def max_previous_correlations(features: torch.Tensor, ranked_indices: torch.Tensor, top_n: int) -> torch.Tensor:
    top_indices = ranked_indices[: min(top_n, ranked_indices.numel())]
    standardized = standardized_columns(features[:, top_indices])
    max_corr = torch.zeros((top_indices.numel(),), dtype=torch.float32)
    selected = []
    for offset, _ in enumerate(top_indices.tolist()):
        candidate = standardized[:, offset]
        if not selected:
            max_corr[offset] = 0.0
        else:
            selected_matrix = standardized[:, torch.tensor(selected, dtype=torch.int64)]
            correlations = torch.abs((selected_matrix * candidate.unsqueeze(1)).mean(dim=0))
            max_corr[offset] = correlations.max()
        selected.append(offset)
    return max_corr


def save_fisher_vs_redundancy_scatter(
    path: Path,
    scores: torch.Tensor,
    ranked_indices: torch.Tensor,
    layer_ids: torch.Tensor,
    features: torch.Tensor,
    top_n: int,
) -> None:
    top_indices = ranked_indices[: min(top_n, ranked_indices.numel())]
    redundancy = max_previous_correlations(features, ranked_indices, top_n=top_n).cpu().numpy()
    fisher = scores[top_indices].cpu().numpy()
    layers = layer_ids[top_indices].cpu().numpy() + 1
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(fisher, redundancy, c=layers, cmap="tab10", s=18, alpha=0.75)
    ax.set_title("Fisher Score vs Redundancy")
    ax.set_xlabel("Train Fisher score")
    ax.set_ylabel("Max abs correlation to higher-ranked feature")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Layer")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def separation_matrix(scores_by_pair: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for pair_score, (left, right) in zip(scores_by_pair, class_pairs(num_classes), strict=True):
        matrix[left, right] = pair_score
        matrix[right, left] = pair_score
    return matrix


def save_class_pair_heatmap(path: Path, features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> None:
    pairwise_scores = pairwise_fisher_scores(features, labels, num_classes)
    pair_strength = pairwise_scores.mean(dim=1)
    matrix = separation_matrix(pair_strength, num_classes=num_classes).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="magma")
    ax.set_title("Mean Pairwise Fisher Separation")
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_per_layer_boxplot(path: Path, scores: torch.Tensor, block_slices: list[slice]) -> None:
    data = [scores[feature_slice].cpu().numpy() for feature_slice in block_slices]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(data, showfliers=False)
    ax.set_title("Per-Layer Fisher Score Distributions")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Train Fisher score")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pca_projection(features: torch.Tensor, rank: int = 2) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    u, s, _ = torch.pca_lowrank(centered, q=max(rank, 4))
    return u[:, :rank] * s[:rank]


def save_projection_figure(
    path: Path,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    samples_per_split: int,
) -> None:
    per_class = max(1, samples_per_split // 10)
    train_subset = stratified_subset_indices(train_labels, per_class)
    test_subset = stratified_subset_indices(test_labels, per_class)

    combined_features = torch.cat((train_features[train_subset], test_features[test_subset]), dim=0)
    combined_labels = torch.cat((train_labels[train_subset], test_labels[test_subset]), dim=0)
    split_flags = torch.cat(
        (
            torch.zeros((train_subset.numel(),), dtype=torch.int64),
            torch.ones((test_subset.numel(),), dtype=torch.int64),
        ),
        dim=0,
    )
    combined_standardized = standardized_columns(combined_features).cpu()
    pca_points = pca_projection(combined_standardized).numpy()
    umap_points = umap.UMAP(n_components=2, random_state=0).fit_transform(combined_standardized.numpy())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, points, title in zip(axes, (pca_points, umap_points), ("PCA", "UMAP"), strict=True):
        for split_value, marker, split_name in ((0, "o", "train"), (1, "x", "test")):
            mask = split_flags == split_value
            scatter = axis.scatter(
                points[mask.numpy(), 0],
                points[mask.numpy(), 1],
                c=combined_labels[mask].numpy(),
                cmap="tab10",
                s=12,
                alpha=0.75,
                marker=marker,
                linewidths=0.4,
            )
        axis.set_title(title)
        axis.set_xlabel("component 1")
        axis.set_ylabel("component 2")
    fig.colorbar(scatter, ax=axes, fraction=0.03, pad=0.04)
    fig.suptitle("Feature-Space Example Projections")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_train_test_stability(path: Path, train_scores: torch.Tensor, test_scores: torch.Tensor, layer_ids: torch.Tensor) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        train_scores.cpu().numpy(),
        test_scores.cpu().numpy(),
        c=(layer_ids.cpu().numpy() + 1),
        cmap="tab10",
        s=12,
        alpha=0.7,
    )
    max_score = float(max(train_scores.max().item(), test_scores.max().item()))
    ax.plot([0.0, max_score], [0.0, max_score], linestyle="--", color="black", linewidth=1.0)
    ax.set_title("Train vs Test Fisher Stability")
    ax.set_xlabel("Train Fisher score")
    ax.set_ylabel("Test Fisher score")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Layer")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    device: torch.device,
) -> tuple[nn.Linear, float]:
    head = nn.Linear(train_features.shape[1], 10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, epochs, warmup_epochs=max(1, min(2, epochs)))

    train_loader = build_feature_loader(train_features, train_labels, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = build_feature_loader(test_features, test_labels, batch_size=batch_size, shuffle=False, num_workers=0)
    best_test_acc = 0.0

    for _ in range(epochs):
        head.train(True)
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device, non_blocking=True).to(torch.float32)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = head(batch_features)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

        head.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device, non_blocking=True).to(torch.float32)
                batch_labels = batch_labels.to(device, non_blocking=True)
                predictions = head(batch_features).argmax(dim=1)
                correct += int((predictions == batch_labels).sum().item())
                total += int(batch_labels.numel())
        best_test_acc = max(best_test_acc, correct / max(1, total))

    return head.to("cpu"), best_test_acc


def save_head_weight_figure(
    path: Path,
    head: nn.Linear,
    ranked_indices: torch.Tensor,
    layer_ids: torch.Tensor,
    top_n: int,
) -> None:
    weights = head.weight.detach().to(torch.float32)
    feature_weight_mass = weights.abs().sum(dim=0)
    top_weight_indices = torch.argsort(feature_weight_mass, descending=True)[: min(top_n, feature_weight_mass.numel())]
    top_weights = weights[:, top_weight_indices].cpu().numpy()

    layer_mass = []
    max_layer = int(layer_ids.max().item()) + 1
    for layer_index in range(max_layer):
        mask = layer_ids == layer_index
        layer_mass.append(float(feature_weight_mass[mask].sum().item()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    heatmap = axes[0].imshow(top_weights, cmap="coolwarm", aspect="auto")
    axes[0].set_title("Linear Head Weights")
    axes[0].set_xlabel("Top abs-weight features")
    axes[0].set_ylabel("Class")
    fig.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].bar(np.arange(1, max_layer + 1), layer_mass)
    axes[1].set_title("Head Weight Mass by Layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Sum abs weights")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summary_payload(
    train_scores: torch.Tensor,
    test_scores: torch.Tensor,
    layer_ids: torch.Tensor,
    block_slices: list[slice],
    linear_probe_acc: float,
) -> dict:
    return {
        "num_selected_features": int(train_scores.numel()),
        "linear_probe_best_test_accuracy": linear_probe_acc,
        "mean_train_fisher": float(train_scores.mean().item()),
        "mean_test_fisher": float(test_scores.mean().item()),
        "mean_train_test_gap": float((train_scores - test_scores).mean().item()),
        "per_layer": [
            {
                "layer": layer_index + 1,
                "feature_count": int(feature_slice.stop - feature_slice.start),
                "mean_train_fisher": float(train_scores[feature_slice].mean().item()),
                "mean_test_fisher": float(test_scores[feature_slice].mean().item()),
                "mean_gap": float((train_scores[feature_slice] - test_scores[feature_slice]).mean().item()),
            }
            for layer_index, feature_slice in enumerate(block_slices)
        ],
        "top_features": [
            {
                "feature_index": int(index.item()),
                "layer": int(layer_ids[index].item()) + 1,
                "train_fisher": float(train_scores[index].item()),
                "test_fisher": float(test_scores[index].item()),
            }
            for index in torch.argsort(train_scores, descending=True)[:20]
        ],
    }


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = output_subdir(args.output_dir, args.selection_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    selection_payload = load_layerwise_selection(args.selection_path)
    train_features, train_labels, test_features, test_labels = load_or_extract_features(
        selection_payload,
        train_cache=args.train_cache,
        test_cache=args.test_cache,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    block_slices = feature_block_slices(selection_payload["layer_selections"])
    layer_ids = feature_layer_ids(block_slices)
    family_masks = expand_feature_family_masks(selection_payload["layer_selections"])
    summaries = {}

    for family_name, family_bit in family_definitions(family_masks):
        if family_bit is None:
            family_mask = torch.ones((train_features.shape[1],), dtype=torch.bool)
        else:
            family_mask = (family_masks & family_bit) != 0
            if int(family_mask.sum().item()) == 0:
                continue

        family_train = train_features[:, family_mask]
        family_test = test_features[:, family_mask]
        family_layer_ids = layer_ids[family_mask]
        family_block_slices = feature_block_slices_from_layer_ids(
            family_layer_ids,
            num_layers=len(selection_payload["layer_selections"]),
        )
        train_scores = fisher_scores(family_train, train_labels, args.num_classes)
        test_scores = fisher_scores(family_test, test_labels, args.num_classes)
        ranked_indices = torch.argsort(train_scores, descending=True)

        save_correlation_heatmap(
            output_dir / f"01_feature_correlation_heatmap_{family_name}.png",
            family_train,
            ranked_indices,
            top_n=args.top_correlation_features,
        )
        save_fisher_vs_redundancy_scatter(
            output_dir / f"02_fisher_vs_redundancy_scatter_{family_name}.png",
            train_scores,
            ranked_indices,
            family_layer_ids,
            family_train,
            top_n=args.top_scatter_features,
        )
        save_class_pair_heatmap(
            output_dir / f"03_class_pair_separation_heatmap_{family_name}.png",
            family_train,
            train_labels,
            num_classes=args.num_classes,
        )
        save_per_layer_boxplot(
            output_dir / f"04_per_layer_separation_curves_{family_name}.png",
            train_scores,
            family_block_slices,
        )

        train_standardized, test_standardized = standardize_train_test(family_train, family_test)
        save_projection_figure(
            output_dir / f"05_pca_umap_projection_{family_name}.png",
            train_standardized,
            train_labels,
            test_standardized,
            test_labels,
            samples_per_split=args.projection_samples_per_split,
        )
        save_train_test_stability(
            output_dir / f"06_train_test_stability_scatter_{family_name}.png",
            train_scores,
            test_scores,
            family_layer_ids,
        )
        head, best_probe_acc = train_linear_probe(
            train_standardized,
            train_labels,
            test_standardized,
            test_labels,
            batch_size=args.head_batch_size,
            epochs=args.head_epochs,
            learning_rate=args.head_learning_rate,
            weight_decay=args.head_weight_decay,
            label_smoothing=args.head_label_smoothing,
            device=device,
        )
        save_head_weight_figure(
            output_dir / f"07_head_weight_analysis_{family_name}.png",
            head,
            ranked_indices,
            family_layer_ids,
            top_n=args.top_correlation_features,
        )
        summaries[family_name] = summary_payload(
            train_scores,
            test_scores,
            family_layer_ids,
            family_block_slices,
            linear_probe_acc=best_probe_acc,
        )

    write_summary(output_dir / "analysis_summary.json", summaries)

    combined_summary = summaries.get("combined", {})
    print(f"saved_analysis_dir={output_dir}")
    print(
        f"selected_features={combined_summary.get('num_selected_features', train_features.shape[1])} "
        f"linear_probe_best_test_acc={combined_summary.get('linear_probe_best_test_accuracy', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
