import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


def _stub(monkeypatch, name, package=False, **attributes):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_temporal_attention(monkeypatch):
    """Load the module while replacing its optional simulator dependencies."""
    package_name = "_uniad_temporal_attention_test"
    source_dir = Path(__file__).resolve().parents[1] / "projects/mmdet3d_plugin/uniad/modules"
    package = _stub(monkeypatch, package_name, package=True)
    package.__path__ = [str(source_dir)]
    _stub(
        monkeypatch,
        f"{package_name}.multi_scale_deformable_attn_function",
        MultiScaleDeformableAttnFunction_fp32=object,
    )

    _stub(monkeypatch, "mmcv", package=True)
    _stub(monkeypatch, "mmcv.ops", package=True)
    _stub(
        monkeypatch,
        "mmcv.ops.multi_scale_deform_attn",
        multi_scale_deformable_attn_pytorch=lambda *args, **kwargs: None,
    )
    _stub(
        monkeypatch,
        "mmcv.cnn",
        xavier_init=lambda *args, **kwargs: None,
        constant_init=lambda *args, **kwargs: None,
    )
    _stub(monkeypatch, "mmcv.cnn.bricks", package=True)

    class Registry:
        def register_module(self):
            return lambda cls: cls

    _stub(monkeypatch, "mmcv.cnn.bricks.registry", ATTENTION=Registry())
    _stub(monkeypatch, "mmcv.runner", package=True)

    class BaseModule(nn.Module):
        def __init__(self, init_cfg=None):
            super().__init__()

    _stub(
        monkeypatch,
        "mmcv.runner.base_module",
        BaseModule=BaseModule,
        ModuleList=nn.ModuleList,
        Sequential=nn.Sequential,
    )
    _stub(monkeypatch, "mmcv.utils", package=True)
    ext_loader = _stub(
        monkeypatch,
        "mmcv.utils.ext_loader",
        load_ext=lambda *args, **kwargs: types.SimpleNamespace(),
    )
    utils = sys.modules["mmcv.utils"]
    utils.ConfigDict = dict
    utils.build_from_cfg = lambda *args, **kwargs: None
    utils.deprecated_api_warning = lambda *args, **kwargs: None
    utils.to_2tuple = lambda value: (value, value)
    utils.ext_loader = ext_loader

    module_name = f"{package_name}.temporal_self_attention"
    spec = importlib.util.spec_from_file_location(
        module_name, source_dir / "temporal_self_attention.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_previous_bev_selection_preserves_sample_alignment(monkeypatch):
    temporal_attention = _load_temporal_attention(monkeypatch)
    value = torch.tensor(
        [
            [[10.0, 10.1]],  # sample 0, previous frame
            [[20.0, 20.1]],  # sample 0, current frame
            [[30.0, 30.1]],  # sample 1, previous frame
            [[40.0, 40.1]],  # sample 1, current frame
        ]
    )

    previous = temporal_attention._select_previous_bev(value, 2, 2)

    torch.testing.assert_close(previous, value[[0, 2]])
    assert not torch.equal(previous, value[:2])


def test_temporal_attention_uses_each_sample_history(monkeypatch):
    temporal_attention = _load_temporal_attention(monkeypatch)
    attention = temporal_attention.TemporalSelfAttention(
        embed_dims=2, num_heads=1, num_levels=1, num_points=1, dropout=0.0
    )

    class CaptureProjection(nn.Module):
        def __init__(self, output_size, capture=False):
            super().__init__()
            self.output_size = output_size
            self.capture = capture

        def forward(self, input_tensor):
            if self.capture:
                self.input = input_tensor.detach().clone()
            return input_tensor.new_zeros(
                input_tensor.shape[0], input_tensor.shape[1], self.output_size
            )

    offsets = CaptureProjection(4, capture=True)
    attention.sampling_offsets = offsets
    attention.attention_weights = CaptureProjection(2)
    attention.value_proj = nn.Identity()
    attention.output_proj = nn.Identity()
    attention.dropout = nn.Identity()
    monkeypatch.setattr(
        temporal_attention,
        "multi_scale_deformable_attn_pytorch",
        lambda value, spatial_shapes, sampling_locations, attention_weights: value.new_zeros(
            value.shape[0], sampling_locations.shape[1], value.shape[-2] * value.shape[-1]
        ),
    )

    attention(
        query=torch.tensor([[[100.0, 101.0]], [[200.0, 201.0]]]),
        value=torch.tensor(
            [
                [[10.0, 11.0]],  # sample 0, previous frame
                [[20.0, 21.0]],  # sample 0, current frame
                [[30.0, 31.0]],  # sample 1, previous frame
                [[40.0, 41.0]],  # sample 1, current frame
            ]
        ),
        reference_points=torch.zeros(4, 1, 1, 2),
        spatial_shapes=torch.tensor([[1, 1]]),
        level_start_index=torch.tensor([0]),
    )

    torch.testing.assert_close(
        offsets.input,
        torch.tensor(
            [
                [[10.0, 11.0, 100.0, 101.0]],
                [[30.0, 31.0, 200.0, 201.0]],
            ]
        ),
    )


def test_previous_bev_selection_rejects_incomplete_queue(monkeypatch):
    temporal_attention = _load_temporal_attention(monkeypatch)
    with pytest.raises(ValueError, match=r"batch_size \* num_bev_queue"):
        temporal_attention._select_previous_bev(torch.zeros(3, 1, 4), 2, 2)
