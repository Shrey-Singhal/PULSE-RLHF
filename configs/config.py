# This config file defines all the model choices, datasets, and hyperparameters for this experiment.

from dataclasses import dataclass


@dataclass
class PulseConfig:

    pretrain_model: str = "trl-lib/pythia-1b-deduped-tldr-sft"

    prompt_data: str = "lkevinzc/tldr-with-sft-reference"

    budget = 50000

    #vllm configs
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True

    top_k: int = -1
    top_p: float = 1.0
    num_samples: int = 6
    max_tokens: int = 512
    temperature: float = 1.0

