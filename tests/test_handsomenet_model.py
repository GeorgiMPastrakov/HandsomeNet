import torch

from handsomenet.models.backbone import HandsomeNetBackbone
from handsomenet.models.graph_attention import GraphAttentionLayer, build_adjacency_mask
from handsomenet.models.handsomenet import HandsomeNet
from handsomenet.models.token_extractor import JointTokenExtractor


def test_backbone_output_shape() -> None:
    model = HandsomeNetBackbone()
    images = torch.randn(2, 3, 224, 224)

    outputs = model(images)

    assert outputs.shape == (2, 192, 14, 14)


def test_token_extractor_output_shape() -> None:
    extractor = JointTokenExtractor()
    feature_map = torch.randn(2, 192, 14, 14)

    outputs = extractor(feature_map)

    assert outputs.shape == (2, 21, 128)


def test_graph_attention_layer_preserves_shape() -> None:
    layer = GraphAttentionLayer()
    tokens = torch.randn(2, 21, 128)

    outputs = layer(tokens)

    assert outputs.shape == (2, 21, 128)


def test_graph_adjacency_mask_includes_self_and_neighbors() -> None:
    adjacency = build_adjacency_mask()

    assert adjacency.shape == (21, 21)
    assert adjacency[0, 0]
    assert adjacency[0, 1]
    assert adjacency[1, 0]
    assert not adjacency[1, 8]


def test_handsomenet_forward_shape_and_range() -> None:
    model = HandsomeNet()
    images = torch.randn(2, 3, 224, 224)

    outputs = model(images)

    assert outputs.shape == (2, 21, 2)
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)
