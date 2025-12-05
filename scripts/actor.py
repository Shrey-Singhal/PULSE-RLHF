from typing import List, cast
import vllm
import torch
from configs.config import PulseConfig

class Actor:

    def __init__(self, config: PulseConfig):
        self.config = config

        self.llm = vllm.LLM(
            model = self.config.pretrain_model,
            tokenizer = self.config.pretrain_model,
            trust_remote_code = self.config.trust_remote_code,
            gpu_memory_utilization = self.config.gpu_memory_utilization
        )

        self.sampling_params = vllm.SamplingParams(
            max_tokens = self.config.max_tokens,
            top_p = self.config.top_p,
            top_k= self.config.top_k,
            temperature = self.config.temperature,
            n = self.config.num_samples
        )

        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts: List[str], sampling_params: vllm.SamplingParams):
        """
        Generate n responses for each prompt.
        Returns a dictionary of responses.:
            {0: [resp1, resp2, ...], 1: [...], ...}
        """

        # strip BOS token
        if self.tokenizer.bos_token:
            prompts = [
                p.removeprefix(self.tokenizer.bos_token)
                for p in prompts
            ]

        outputs = self.llm.generate(
            prompts,
            sampling_params,
            use_tqdm=False
        )

        candidates = {}
        for i, out in enumerate(outputs):
            candidates[i] = []
            for n in range(sampling_params.n):
                text = out.outputs[n].text.strip()
                candidates[i].append(text)

        return candidates
