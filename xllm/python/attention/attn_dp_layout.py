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

"""Reference layouts for the three-collective GLM attention DP/TP path.

The target path keeps ``q_a_proj`` and ``kv_a_proj_with_mqa`` replicated while
sharding the head-dependent weights. This module describes the logical tensor
layouts at C2 (Q heads), C3 (attention latent), and C4 (O reduction). It is a
CPU-testable shape contract and deliberately contains no process-group calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttnDpLayout:
    """Shape contract for one attention DP/TP group."""

    world_size: int
    num_heads: int
    q_head_dim: int
    kv_lora_rank: int
    hidden_size: int
    owner_heads: int | None = None

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if self.num_heads <= 0 or self.num_heads % self.world_size:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by world_size ({self.world_size})")
        if self.q_head_dim <= 0 or self.kv_lora_rank <= 0 or self.hidden_size <= 0:
            raise ValueError("attention dimensions must be positive")
        if self.owner_heads is not None and self.owner_heads <= 0:
            raise ValueError("owner_heads must be positive")

    @property
    def local_heads(self) -> int:
        return self.num_heads // self.world_size

    @property
    def owner_local_heads(self) -> int:
        """Heads executed by a TP token-owner rank.

        The original reference path uses one rank for both ownership and
        weight sharding, so this equals ``local_heads``.  In DP2/TP8 the
        candidate has 16 weight ranks but only 8 token-owner ranks per DP
        replica, hence each owner receives two weight shards.
        """
        return self.owner_heads if self.owner_heads is not None else self.local_heads


__all__ = ["AttnDpLayout"]
