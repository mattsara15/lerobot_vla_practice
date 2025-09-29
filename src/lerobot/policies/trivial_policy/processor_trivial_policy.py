#!/usr/bin/env python

"""Processor pipeline scaffolding for the trivial policy.

This module mirrors the structure of the diffusion policy processors but leaves
the implementations intentionally blank. It's intended as a skeleton for later
development.
"""

from typing import Any, Tuple

import torch

from lerobot.processor import (
	AddBatchDimensionProcessorStep,
	DeviceProcessorStep,
	NormalizerProcessorStep,
	PolicyAction,
	PolicyProcessorPipeline,
	RenameObservationsProcessorStep,
	UnnormalizerProcessorStep,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.policies.trivial_policy.configuration_trivial_policy import TrivialConfig


def make_trivial_pre_post_processors(
	config: TrivialConfig,
	dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> Tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]:
	"""Construct pre- and post-processors for the trivial policy.

	All steps are declared but not implemented; this function returns pipelines
	with the expected steps present. Implementation of step behaviour is left
	for future work.
	"""

	input_steps = [
		RenameObservationsProcessorStep(rename_map={}),
		AddBatchDimensionProcessorStep(),
		DeviceProcessorStep(device=config.device),
		NormalizerProcessorStep(features={}, norm_map={}, stats=dataset_stats),
	]
	output_steps = [
		UnnormalizerProcessorStep(features={}, norm_map={}, stats=dataset_stats),
		DeviceProcessorStep(device="cpu"),
	]

	pre = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
		steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME
	)
	post = PolicyProcessorPipeline[PolicyAction, PolicyAction](
		steps=output_steps, name=POLICY_POSTPROCESSOR_DEFAULT_NAME
	)

	return pre, post

