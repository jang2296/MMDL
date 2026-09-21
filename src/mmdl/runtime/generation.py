"""Generated-only presence penalty, before temperature and probability filters."""

import torch
from transformers import LogitsProcessor


class GeneratedOnlyPresencePenalty(LogitsProcessor):
    def __init__(self, prompt_length, penalty):
        if prompt_length < 0 or penalty < 0:
            raise ValueError("Invalid prompt length or penalty")
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if generated.numel() == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return scores - seen.to(scores.dtype) * self.penalty
