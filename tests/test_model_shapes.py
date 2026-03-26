import torch

from random_feature_classifier.config import ModelConfig
from random_feature_classifier.model import make_model


def test_same_seed_gives_same_outputs() -> None:
    config = ModelConfig(seed=7, backbone_kind="strict_ones", head_kind="linear", width_multiplier=1)
    model_a = make_model(config).eval()
    model_b = make_model(config).eval()
    inputs = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        outputs_a = model_a(inputs)
        outputs_b = model_b(inputs)

    assert torch.allclose(outputs_a, outputs_b)


def test_backbones_are_swappable() -> None:
    strict_model = make_model(ModelConfig(backbone_kind="strict_ones", head_kind="linear", width_multiplier=1))
    random_model = make_model(ModelConfig(backbone_kind="random_projection", head_kind="linear", width_multiplier=1))
    inputs = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        strict_features = strict_model.forward_features(inputs)
        random_features = random_model.forward_features(inputs)

    assert strict_features.shape == random_features.shape


def test_scalar_head_supports_wider_features() -> None:
    config = ModelConfig(
        backbone_kind="strict_ones",
        head_kind="scalar",
        width_multiplier=3,
        global_feature_dim=2048,
    )
    model = make_model(config).eval()
    inputs = torch.randn(3, 3, 32, 32)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (3, config.num_classes)


def test_tapped_stage_dimensions_accumulate() -> None:
    config = ModelConfig(
        backbone_kind="strict_ones",
        head_kind="linear",
        width_multiplier=1,
        tap_stages=(2, 4),
        global_feature_dim=0,
    )
    model = make_model(config).eval()
    inputs = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        features = model.forward_features(inputs)

    assert features.shape == (2, 128 + 512)


def test_tapped_pooled_features_stay_finite_in_float16() -> None:
    config = ModelConfig(
        backbone_kind="strict_ones",
        head_kind="linear",
        width_multiplier=2,
        tap_stages=(1, 2, 3, 4),
        global_feature_dim=0,
    )
    model = make_model(config).eval()
    inputs = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        features = model.forward_features(inputs).to(torch.float16)

    assert torch.isfinite(features).all()
