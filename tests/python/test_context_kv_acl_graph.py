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

from typing import cast
from unittest.mock import Mock

import pytest
import torch

from xllm.python.attention.backend import AttentionBackend, LayerCache
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("default_stream", [True, False], ids=["default_startup", "nondefault_startup"])
def test_real_context_graph_matches_eager_and_preserves_masked_slots(
    default_stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xllm.python.models.qwen3_dspark import Qwen3DSparkForCausalLM

    monkeypatch.setattr("xllm.python.model_executor.runners.acl_graph._GRAPH_POOL", None)
    torch.manual_seed(23)
    model = Qwen3DSparkForCausalLM(
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "intermediate_size": 128,
            "max_position_embeddings": 32,
            "vocab_size": 8,
            "markov_rank": 2,
            "dtype": "bfloat16",
            "device": "npu:0",
        }
    )
    with torch.no_grad():
        model.model.fc.load_weight(torch.randn(64, 128), tp_rank=0)
        for parameter in model.parameters():
            parameter.normal_()
        model.model._build_context_kv_buffers()
    caches = [
        (
            torch.full((1, 128, 1, 64), 42.0, dtype=torch.bfloat16, device="npu:0"),
            torch.full((1, 128, 1, 64), 43.0, dtype=torch.bfloat16, device="npu:0"),
        )
        for _ in range(2)
    ]
    graph_runner = DecodeAclGraphRunner(
        model.model,
        cast("AttentionBackend", Mock(spec=AttentionBackend)),
        torch.device("npu:0"),
        4,
        32,
        num_speculative_tokens=2,
        is_spec_draft=True,
    )
    graph_runner.bind_layer_caches([LayerCache(*cache) for cache in caches])
    stream = torch.npu.default_stream("npu:0") if default_stream else torch.npu.Stream(device="npu:0")
    if stream != torch.npu.current_stream():
        stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph_runner.warmup_context_kv(1)
        assert torch.npu.current_stream() == stream
        for keys, values in caches:
            assert torch.all(keys == 42).item()
            assert torch.all(values == 43).item()
        for iteration, rows in enumerate((4, 3, 4, 1, 3, 2)):
            hidden = torch.randn(rows, 128, dtype=torch.bfloat16, device="npu:0")
            positions = torch.arange(iteration, iteration + rows, dtype=torch.int32, device="npu:0")
            slots = torch.arange(2, rows + 2, dtype=torch.int32, device="npu:0")
            slots[-1] = -1
            reference_caches = [(key.clone(), value.clone()) for key, value in caches]
            reference = model.model.write_context_kv(
                hidden, positions, slots, [LayerCache(*cache) for cache in reference_caches], None
            )
            actual = graph_runner.replay_context_kv(hidden, positions, slots)
            torch.npu.synchronize()
            torch.testing.assert_close(actual, reference)
            for actual_pair, expected_pair in zip(caches, reference_caches):
                for actual_cache, expected_cache in zip(actual_pair, expected_pair):
                    torch.testing.assert_close(actual_cache, expected_cache)
