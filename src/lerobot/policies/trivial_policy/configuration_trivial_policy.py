
#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("trivial")
@dataclass
class TrivialConfig(PreTrainedConfig):
	"""Configuration placeholder for the trivial policy.

	Fields are intentionally minimal. Real implementations should expand this
	with feature definitions, normalization mappings, device, and other fields
	mirroring more complete policy configs.
	"""

	device: str = "cpu"

