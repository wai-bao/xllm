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

"""Fake attention backend and model shared by the embedded-Python C++ tests."""

from __future__ import annotations

from contextlib import AbstractContextManager
from unittest.mock import patch

import torch

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata, LayerCache
from xllm.python.layers.attention import Attention
from xllm.python.model_executor import executor as executor_module


class FakeBackend(AttentionBackend):
    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        pass

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        pass

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        return q

    @property
    def num_kv_blocks(self) -> int:
        return 0

    @property
    def page_size(self) -> int:
        return 1


class FakeModel(torch.nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, device=device))
        self.attention = Attention(1, 1, 8, 1.0, 0, 0)
        self.model = torch.nn.Identity()


def patched_fake_backend() -> AbstractContextManager[object]:
    """Context manager that forces ModelExecutor to build FakeBackend."""
    return patch.object(executor_module, "_create_attention_backend", return_value=FakeBackend())
