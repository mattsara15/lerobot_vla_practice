#!/usr/bin/env python

"""Trivial policy model scaffolding.

This file mirrors the structure of the diffusion policy model file but leaves
methods intentionally blank. It's a placeholder for future trivial policy
implementations.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn, Tensor

from lerobot.policies.trivial_policy.configuration_trivial_policy import TrivialConfig
from lerobot.policies.pretrained import PreTrainedPolicy


class TrivialPolicy(PreTrainedPolicy):
    """A placeholder TrivialPolicy.

    Methods are present but left unimplemented (pass) to serve as a scaffold.
    """

    config_class = TrivialConfig
    name = "trivial"

    def __init__(self, config: TrivialConfig):
        # Intentionally left blank for future implementation
        pass

    def get_optim_params(self) -> dict[str, Any]:
        # Intentionally left blank
        pass

    def reset(self):
        # Intentionally left blank
        pass

    @torch.no_grad()
    def act(self, observations: dict[str, Tensor]) -> Tensor:
        # Intentionally left blank
        pass


class TrivialModel(nn.Module):
    """Simple model placeholder for the trivial policy."""

    def __init__(self, config: TrivialConfig):
        super().__init__()
        # Intentionally left blank
        pass

    def forward(self, x: Tensor) -> Tensor:
        # Intentionally left blank
        pass
