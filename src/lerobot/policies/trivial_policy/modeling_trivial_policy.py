#!/usr/bin/env python

"""Trivial policy model scaffolding.

This file mirrors the structure of the diffusion policy model file but leaves
methods intentionally blank. It's a placeholder for future trivial policy
implementations.
"""

from __future__ import annotations

from typing import Any

from lerobot.utils.constants import OBS_STATE, ACTION, OBS_IMAGES
import torch
from torch import nn, Tensor

from lerobot.policies.trivial_policy.configuration_trivial_policy import TrivialConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from transformers import AutoModelForImageTextToText, AutoProcessor

class TrivialModel(nn.Module):
    """Standalone nn.Module for the trivial policy model.

    Responsibilities:
    - load the underlying transformer and tokenizer
    - expose a simple `generate_numbers` method that returns numeric actions
    """

    def __init__(self, model_path: str = "OpenGVLab/InternVL3-1B-hf") -> None:
        super().__init__()
        self.model_path = model_path
        # AutoModel is not strictly an nn.Module wrapper for chat-based models, but
        # we store it here for convenience and lifecycle management.
        self._transformer = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

        # Tokenizer is not a nn.Module, but belongs here logically.
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    def generate_numbers(self, prompt: str, max_new_tokens: int = 256, do_sample: bool = True) -> list[float]:
        """Generate a list of floats from the underlying chat model.

        This method keeps the same high-level behavior as the previous inline
        usage: call the model.chat API with a prompt and parse a comma-separated
        list of numbers from the response. The implementation is intentionally
        tolerant of parsing errors and returns an empty list on failure.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return a list of 10 numbers as a comma-separated string"},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to('mps', dtype=torch.bfloat16)
        generate_ids = self._transformer.generate(**inputs, max_new_tokens=50)
        response = self._processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


        print(f"LLM Response: {response}")

        # Parse a comma separated list of numbers
        numbers: list[float] = []
        for token in response.split(','):
            token = token.strip()
            if token == "":
                continue
            try:
                numbers.append(float(token))
            except ValueError:
                # ignore tokens that don't parse
                continue

        return numbers


class TrivialPolicy(PreTrainedPolicy):
    """A placeholder TrivialPolicy that uses TrivialModel.

    The policy retains the original scaffold: training methods remain
    placeholders. The heavy-lifting model logic has been moved into
    `TrivialModel` so it can be unit tested independently and used as an
    nn.Module where appropriate.
    """

    config_class = TrivialConfig
    name = "trivial"

    def __init__(self, config: TrivialConfig):
        super().__init__(config)
        # create a standalone model instance (kept on CPU by default here)
        self._model = TrivialModel()
        # simple single-layer head for training/evaluation in forward
        # we'll lazily initialize it on the first forward if needed
        self._head: nn.Module | None = None

        self._queues = None

    def get_optim_params(self) -> dict[str, Any]:
        # Intentionally left blank; training not implemented for trivial policy.
        return {}

    def reset(self):
        # Intentionally left blank
        pass

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate actions for the batch using the TrivialModel.

        The prior implementation attempted to build a response and then
        slice into a sequence of actions. Here we mimic that: ask the model
        for a CSV of numbers and return a Tensor shaped appropriately.
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        prompt = "Return a comma separated list of 10 randomly picked numbers"
        numbers = self._model.generate_numbers(prompt, max_new_tokens=1024, do_sample=True)

        # If parsing failed, return zeros
        if len(numbers) == 0:
            numbers = [0.0] * (self.config.n_action_steps)

        # Create a tensor shaped (batch_size, n_action_steps, action_dim?)
        # The original code was unclear about action dimensionality; assume 1D
        actions = torch.tensor(numbers[: self.config.n_action_steps], dtype=torch.float32)
        actions = actions.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)

        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # For the trivial policy we simply delegate to generate_actions
        actions = self.generate_actions(batch)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        actions = self.predict_action_chunk(batch)
        print(f"Found {len(actions)} actions")
        return actions[0]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        pass
