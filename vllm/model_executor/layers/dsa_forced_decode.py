# SPDX-License-Identifier: Apache-2.0
"""tollbooth: recorded-continuation (teacher-forced) decode as a V1 logits processor.

Each request carries ``SamplingParams.extra_args["forced_tokens"]``: the exact token
sequence the model must emit. At every decode step the processor masks the logits of that
request to -inf everywhere except the next forced token, so greedy sampling emits it, the
engine appends it to the request and advances the KV cache through the normal decode path
one token per step. Nothing is pre-filled from the continuation; nothing is replaced after
the fact. The model's own preferred token is irrelevant.

Registration: ``LLM(..., logits_processors=["vllm.model_executor.layers.dsa_forced_decode.ForcedSequenceLogitsProcessor"])``.

Use ``max_tokens=len(forced_tokens)`` and ``ignore_eos=True`` so a forced end-of-sentence
token inside the sequence does not stop generation early; ``temperature=0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams


class ForcedSequenceLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        self.device = device
        # batch index -> (forced token ids, running output token ids of the request)
        self.state: dict[int, tuple[list[int], list[int]]] = {}
        self.rows = torch.empty(0, dtype=torch.int64, device=device)
        self.cols = torch.empty(0, dtype=torch.int64, device=device)
        self.overrun: dict[int, int] = {}  # index -> steps requested beyond the forced sequence

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams"):
        ft = (sampling_params.extra_args or {}).get("forced_tokens")
        if ft is None:
            return None
        if not isinstance(ft, (list, tuple)) or not all(isinstance(t, int) and t >= 0 for t in ft):
            raise ValueError("forced_tokens must be a list of non-negative ints")
        if sampling_params.max_tokens is not None and sampling_params.max_tokens > len(ft):
            raise ValueError("max_tokens exceeds len(forced_tokens); the forced sequence would be overrun")
        return None

    def is_argmax_invariant(self) -> bool:
        return False

    @staticmethod
    def add_request(params: "SamplingParams", _prompt: list[int] | None, output_tok_ids: list[int]):
        ft = (params.extra_args or {}).get("forced_tokens")
        if not ft:
            return None
        return (list(ft), output_tok_ids)

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.state, batch_update, self.add_request)
        rows, cols = [], []
        for index, (forced, out) in self.state.items():
            step = len(out)  # tokens already emitted == next forced position
            if step < len(forced):
                rows.append(index)
                cols.append(forced[step])
            else:
                self.overrun[index] = self.overrun.get(index, 0) + 1
        self.rows = torch.tensor(rows, dtype=torch.int64, device=self.device)
        self.cols = torch.tensor(cols, dtype=torch.int64, device=self.device)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.rows.numel() == 0:
            return logits
        keep = logits[self.rows, self.cols].clone()
        logits[self.rows] = float("-inf")
        logits[self.rows, self.cols] = torch.where(torch.isfinite(keep), keep, torch.zeros_like(keep))
        return logits
