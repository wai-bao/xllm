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

"""Shape and TP contracts for Phi-3's already fused checkpoint weights.

Source matrices use [output, input] order: Q then K then V, or gate then up.
The production Phi-3 loader and checkpoint tests share this contract. It does
not validate attention semantics or support KV head replication.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from xllm.python.model_loader.sharding import shard_tensor


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class Phi3CheckpointConfig:
    """Structural dimensions needed by the two fused weight contracts."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    num_hidden_layers: int

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "num_hidden_layers",
        ):
            _positive_int(getattr(self, name), name)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def parse_phi3_config(config: Mapping[str, object]) -> Phi3CheckpointConfig:
    """Validate structural fields; absent/null KV heads default to query heads."""
    if not isinstance(config, Mapping):
        raise ValueError("config must be a JSON object")
    if config.get("model_type") != "phi3":
        raise ValueError(f"model_type must be 'phi3', got {config.get('model_type')!r}")
    heads = _positive_int(config.get("num_attention_heads"), "num_attention_heads")
    kv_heads = config.get("num_key_value_heads")
    dimensions = Phi3CheckpointConfig(
        hidden_size=_positive_int(config.get("hidden_size"), "hidden_size"),
        num_attention_heads=heads,
        num_key_value_heads=heads if kv_heads is None else _positive_int(kv_heads, "num_key_value_heads"),
        intermediate_size=_positive_int(config.get("intermediate_size"), "intermediate_size"),
        num_hidden_layers=_positive_int(config.get("num_hidden_layers"), "num_hidden_layers"),
    )
    if "head_dim" in config and _positive_int(config["head_dim"], "head_dim") != dimensions.head_dim:
        raise ValueError(f"head_dim must equal hidden_size / num_attention_heads ({dimensions.head_dim})")
    return dimensions


def expected_phi3_shapes(config: Phi3CheckpointConfig) -> dict[str, tuple[int, int]]:
    """Return global checkpoint shapes for every QKV and gate-up weight."""
    qkv_rows = (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim
    shapes = {}
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        shapes[f"{prefix}.self_attn.qkv_proj.weight"] = (qkv_rows, config.hidden_size)
        shapes[f"{prefix}.mlp.gate_up_proj.weight"] = (2 * config.intermediate_size, config.hidden_size)
    return shapes


def _group_sizes(config: Phi3CheckpointConfig, name: str) -> tuple[int, ...]:
    if name.endswith(".self_attn.qkv_proj.weight"):
        kv_rows = config.num_key_value_heads * config.head_dim
        return (config.hidden_size, kv_rows, kv_rows)
    return (config.intermediate_size, config.intermediate_size)


def _fused_weight_shape(config: Phi3CheckpointConfig, name: str) -> tuple[int, int]:
    match = re.fullmatch(r"model\.layers\.(0|[1-9][0-9]*)\.(self_attn\.qkv_proj|mlp\.gate_up_proj)\.weight", name)
    if match is None or len(match[1]) > len(str(config.num_hidden_layers)) or int(match[1]) >= config.num_hidden_layers:
        raise ValueError(f"{name}: unknown Phi-3 fused weight key")
    return (sum(_group_sizes(config, name)), config.hidden_size)


def _validate_tp(config: Phi3CheckpointConfig, name: str, tp_size: int) -> None:
    _positive_int(tp_size, f"{name}: tp_size")
    if name.endswith(".self_attn.qkv_proj.weight"):
        dimensions = (
            ("num_attention_heads", config.num_attention_heads),
            ("num_key_value_heads", config.num_key_value_heads),
        )
    else:
        dimensions = (("intermediate_size", config.intermediate_size),)
    for field, value in dimensions:
        if value % tp_size:
            raise ValueError(f"{name}: {field}={value} must be divisible by tp_size={tp_size}")


def shard_phi3_fused_weight(
    weight: torch.Tensor,
    *,
    name: str,
    config: Phi3CheckpointConfig,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Shard each logical output group, then concatenate in checkpoint order.

    ``name`` must be a full target key from ``expected_phi3_shapes``. The result
    is contiguous and leaves the source unchanged. At TP=1 an already
    contiguous source is returned directly, avoiding a full-weight copy.
    """
    expected_shape = _fused_weight_shape(config, name)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(f"{name}: expected shape {list(expected_shape)}, got {list(weight.shape)}")
    _validate_tp(config, name, tp_size)
    if isinstance(tp_rank, bool) or not isinstance(tp_rank, int) or not 0 <= tp_rank < tp_size:
        raise ValueError(f"{name}: tp_rank must be an integer in [0, {tp_size}), got {tp_rank!r}")
    if tp_size == 1:
        return weight.contiguous()
    groups = weight.split(_group_sizes(config, name), dim=0)
    return torch.cat([shard_tensor(group, 0, tp_rank, tp_size, name=name, contiguous=False) for group in groups], dim=0)
