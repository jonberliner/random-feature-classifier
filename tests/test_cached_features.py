from pathlib import Path

import torch

from random_feature_classifier.cache import (
    load_feature_cache,
    load_topk_feature_cache,
    save_feature_cache,
    save_topk_feature_cache,
)
from random_feature_classifier.config import ModelConfig
from random_feature_classifier.heads import make_head
from random_feature_classifier.model import make_feature_extractor, make_model


def test_feature_extractor_matches_model_forward_features() -> None:
    config = ModelConfig(backbone_kind="strict_ones", head_kind="linear", width_multiplier=1, global_feature_dim=256)
    extractor = make_feature_extractor(config).eval()
    model = make_model(config).eval()
    inputs = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        extracted = extractor(inputs)
        model_features = model.forward_features(inputs)

    assert torch.allclose(extracted, model_features)


def test_feature_cache_round_trip_and_head_forward(tmp_path: Path) -> None:
    config = ModelConfig(backbone_kind="strict_ones", head_kind="linear", global_feature_dim=128)
    features = torch.randn(8, 128, dtype=torch.float16)
    labels = torch.randint(0, config.num_classes, (8,), dtype=torch.int64)
    cache_path = tmp_path / "train_features.pt"

    save_feature_cache(cache_path, features, labels, config, split="train")
    loaded_features, loaded_labels, metadata = load_feature_cache(str(cache_path))
    head = make_head("linear", loaded_features.shape[1], config)

    with torch.no_grad():
        logits = head(loaded_features.to(torch.float32))

    assert torch.equal(loaded_labels, labels)
    assert metadata["split"] == "train"
    assert logits.shape == (8, config.num_classes)


def test_topk_feature_cache_round_trip(tmp_path: Path) -> None:
    global_features = torch.randn(6, 32, dtype=torch.float16)
    pairwise_features = torch.randn(6, 45, 8, dtype=torch.float16)
    labels = torch.randint(0, 10, (6,), dtype=torch.int64)
    cache_path = tmp_path / "train_topk_features.pt"

    save_topk_feature_cache(
        cache_path,
        global_features=global_features,
        pairwise_features=pairwise_features,
        labels=labels,
        metadata={"seed": 7, "pairwise_per_pair_keeps": [0, 2, 4]},
        split="train",
    )
    loaded_global, loaded_pairwise, loaded_labels, metadata = load_topk_feature_cache(str(cache_path))

    assert torch.equal(loaded_global, global_features)
    assert torch.equal(loaded_pairwise, pairwise_features)
    assert torch.equal(loaded_labels, labels)
    assert metadata["split"] == "train"
    assert metadata["pairwise_per_pair_keeps"] == [0, 2, 4]
