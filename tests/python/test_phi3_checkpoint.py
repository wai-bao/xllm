# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline real-metadata and small-tensor checks for the Phi-3 contract."""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch

from xllm.python.models.phi3_checkpoint import (
    parse_phi3_config,
    shard_phi3_fused_weight,
)

_QKV = "model.layers.0.self_attn.qkv_proj.weight"
_GATE_UP = "model.layers.0.mlp.gate_up_proj.weight"
_CONFIG = {
    "model_type": "phi3",
    "hidden_size": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 12,
    "num_hidden_layers": 1,
}


@pytest.mark.parametrize("field", ["hidden_size", "num_attention_heads", "intermediate_size", "num_hidden_layers"])
def test_missing_required_config_field(field: str) -> None:
    config = {key: value for key, value in _CONFIG.items() if key != field}
    with pytest.raises(ValueError, match=field):
        parse_phi3_config(config)


@pytest.mark.parametrize("field", [key for key in _CONFIG if key != "model_type"])
@pytest.mark.parametrize("value", [0, -1, True, "4", 4.5])
def test_config_rejects_invalid_dimensions(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        parse_phi3_config({**_CONFIG, field: value})


@pytest.mark.parametrize("kv_heads", ["missing", None])
def test_default_kv_heads_matches_reference(kv_heads: object) -> None:
    config = dict(_CONFIG)
    if kv_heads == "missing":
        del config["num_key_value_heads"]
    else:
        config["num_key_value_heads"] = kv_heads
    assert parse_phi3_config(config).num_key_value_heads == 4


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_type": "llama"}, "model_type"),
        ({"hidden_size": 9}, "hidden_size.*num_attention_heads"),
        ({"num_key_value_heads": 3}, "num_attention_heads.*num_key_value_heads"),
        ({"num_key_value_heads": 8}, "num_attention_heads.*num_key_value_heads"),
        ({"head_dim": 4}, "head_dim"),
        ({"head_dim": True}, "head_dim"),
    ],
)
def test_config_rejects_incompatible_structure(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_phi3_config({**_CONFIG, **changes})


@pytest.mark.parametrize(
    ("name", "rank", "rows"),
    [
        (_QKV, 0, [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]),
        (_QKV, 1, [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23]),
        (_GATE_UP, 0, [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]),
        (_GATE_UP, 1, [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]),
    ],
)
def test_shards_have_independent_expected_rows(name: str, rank: int, rows: list[int]) -> None:
    # Distinct groups, rows and columns make reorderings and whole-matrix splits visible.
    source = torch.arange(24, dtype=torch.int64)[:, None] * 100 + torch.arange(8)[None, :]
    original = source.clone()
    result = shard_phi3_fused_weight(source, name=name, config=parse_phi3_config(_CONFIG), tp_rank=rank, tp_size=2)
    expected = torch.tensor(rows)[:, None] * 100 + torch.arange(8)[None, :]
    assert torch.equal(result, expected)
    assert torch.equal(source, original)
    assert result.is_contiguous()
    assert result.data_ptr() != source.data_ptr()


@pytest.mark.parametrize(("name", "group_count", "group_rows"), [(_QKV, 3, 8), (_GATE_UP, 2, 12)])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_reconstructs_weights_and_projection(name: str, group_count: int, group_rows: int, tp_size: int) -> None:
    # Small integer-valued float32 dot products remain exactly representable.
    # Zero tolerance therefore catches any missing, repeated or reordered row.
    source = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
    dimensions = parse_phi3_config(_CONFIG)
    parts = [
        shard_phi3_fused_weight(source, name=name, config=dimensions, tp_rank=rank, tp_size=tp_size)
        for rank in range(tp_size)
    ]
    local_rows = group_rows // tp_size
    reconstructed = torch.cat(
        [
            torch.cat([part[group * local_rows : (group + 1) * local_rows] for part in parts])
            for group in range(group_count)
        ]
    )
    assert torch.equal(reconstructed, source)
    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    outputs = [inputs @ part.T for part in parts]
    projected = torch.cat(
        [
            torch.cat([output[:, group * local_rows : (group + 1) * local_rows] for output in outputs], dim=-1)
            for group in range(group_count)
        ],
        dim=-1,
    )
    torch.testing.assert_close(projected, inputs @ source.T, rtol=0, atol=0)
    if tp_size > 1:
        assert all(part.data_ptr() != source.data_ptr() for part in parts)


def test_grouped_query_attention_preserves_unequal_group_sizes() -> None:
    dimensions = parse_phi3_config({**_CONFIG, "num_key_value_heads": 2})
    source = torch.arange(16 * 8).reshape(16, 8)
    for rank, rows in [(0, [0, 1, 2, 3, 8, 9, 12, 13]), (1, [4, 5, 6, 7, 10, 11, 14, 15])]:
        result = shard_phi3_fused_weight(source, name=_QKV, config=dimensions, tp_rank=rank, tp_size=2)
        assert torch.equal(result, source[rows])


@pytest.mark.parametrize("name", [_QKV, _GATE_UP])
@pytest.mark.parametrize("tp_size", [0, -1, True, 1.5, 5])
def test_sharder_rejects_invalid_tp(name: str, tp_size: Any) -> None:
    with pytest.raises(ValueError, match=re.escape(name) + ".*tp_size"):
        shard_phi3_fused_weight(
            torch.zeros(24, 8), name=name, config=parse_phi3_config(_CONFIG), tp_rank=0, tp_size=tp_size
        )


@pytest.mark.parametrize("name", [_QKV, _GATE_UP])
@pytest.mark.parametrize("rank", [-1, 2, True, 0.5])
def test_sharder_rejects_invalid_rank(name: str, rank: Any) -> None:
    with pytest.raises(ValueError, match=re.escape(name) + ".*tp_rank"):
        shard_phi3_fused_weight(
            torch.zeros(24, 8), name=name, config=parse_phi3_config(_CONFIG), tp_rank=rank, tp_size=2
        )


@pytest.mark.parametrize("name", [_QKV, _GATE_UP])
@pytest.mark.parametrize("shape", [(23, 8), (24, 7), (24, 8, 1)])
def test_sharder_reports_expected_and_actual_shape(name: str, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=re.escape(name) + r": expected shape \[24, 8\], got"):
        shard_phi3_fused_weight(torch.zeros(shape), name=name, config=parse_phi3_config(_CONFIG), tp_rank=0, tp_size=2)


def test_sharder_rejects_unknown_key() -> None:
    name = "model.layers.0.self_attn.q_proj.weight"
    with pytest.raises(ValueError, match=re.escape(name) + ": unknown"):
        shard_phi3_fused_weight(torch.zeros(24, 8), name=name, config=parse_phi3_config(_CONFIG), tp_rank=0, tp_size=1)


@pytest.mark.parametrize("name", [_QKV, _GATE_UP])
def test_tp1_avoids_copying_contiguous_checkpoint_weights(name: str) -> None:
    source = torch.zeros(24, 8)
    result = shard_phi3_fused_weight(source, name=name, config=parse_phi3_config(_CONFIG), tp_rank=0, tp_size=1)
    assert result.data_ptr() == source.data_ptr()


@pytest.mark.parametrize("layer", ["-1", "1", "01", "x", "0.extra", "9" * 5000])
def test_sharder_rejects_nonexistent_or_noncanonical_layer_keys(layer: str) -> None:
    name = f"model.layers.{layer}.self_attn.qkv_proj.weight"
    with pytest.raises(ValueError, match="unknown Phi-3 fused weight key") as error:
        shard_phi3_fused_weight(torch.zeros(24, 8), name=name, config=parse_phi3_config(_CONFIG), tp_rank=0, tp_size=1)
    assert str(error.value).startswith(name + ":")
