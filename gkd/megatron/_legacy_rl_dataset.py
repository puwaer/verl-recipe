# Copyright 2026 (recipe/gkd local compat shim)
"""RLHFDataset subclass that re-adds tokenized keys for the GKD recipe.

New verl's `RLHFDataset.__getitem__` only emits `raw_prompt` (a list of message
dicts) plus auxiliary fields; tokenization has been moved to AgentLoop on the
rollout side. However, GKD's `ray_trainer.py:_async_gen_next_batch` still
expects the old key set (`input_ids`, `attention_mask`, `position_ids` as
tensors plus `raw_prompt_ids` as a non-tensor list). This subclass restores
those keys by tokenizing in `__getitem__`, keeping the GKD recipe unchanged.
"""

from __future__ import annotations

import torch

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import tokenize_and_postprocess_data


class LegacyRLHFDataset(RLHFDataset):
    """Adds tokenized keys (input_ids/attention_mask/position_ids/raw_prompt_ids)
    expected by the GKD recipe's `_async_gen_next_batch`."""

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)

        messages = row_dict["raw_prompt"]

        # Chat-templated prompt text (kept as string for raw_prompt_ids tokenization)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize and left-pad to max_prompt_length, returning (1, L) tensors.
        input_ids, attention_mask = tokenize_and_postprocess_data(
            prompt=prompt_text,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        # Drop the leading batch dim added by tokenize_and_postprocess_data
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = compute_position_id_with_mask(attention_mask[0])

        # `raw_prompt_ids` is the unpadded token list used by vLLM rollout
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
        )

        return row_dict
