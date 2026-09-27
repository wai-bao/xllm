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

"""Shared CPU helpers for GLM-5.2 indexer tests (kernel fakes)."""

from __future__ import annotations

from typing import Any

import torch

from xllm.python.models import glm5_2


def make_indexer(**config_overrides: Any) -> glm5_2.Glm52Indexer:
    """Build a tiny real Glm52Indexer on CPU with the static W8A8 format."""
    values: dict[str, Any] = {
        "hidden_size": 2,
        "q_lora_rank": 2,
        "index_n_heads": 1,
        "index_head_dim": 2,
        "qk_rope_head_dim": 2,
        "index_topk": 1,
        "indexer_rope_interleave": False,
    }
    values.update(config_overrides)
    cfg = glm5_2.Glm52Config(**values)
    indexer = glm5_2.Glm52Indexer(cfg, torch.float32, torch.device("cpu"))
    # Static W8A8 runs on CPU; only its NPU kernels need patching.
    indexer.wq_b._set_dynamic_activation(False)
    return indexer


def quantize_per_tensor(x: torch.Tensor, *_args: object) -> torch.Tensor:
    return x.round().to(torch.int8)


def quant_matmul(x: torch.Tensor, weight: torch.Tensor, *_args: object) -> torch.Tensor:
    return x.float() @ weight.float().T


def dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones_like(x, dtype=torch.int8), torch.ones(x.shape[:-1])


def scatter_nd_update(cache: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> None:
    flat = indices.flatten()
    valid = flat >= 0
    cache.index_copy_(0, flat[valid], values[valid])
