from typing import Dict, List, cast
import vllm
import torch
from configs.config import PulseConfig
from RMbackbone import RMbackbone

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

        self.backbone = RMbackbone(config)

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
    
    def get_features(self, prompts: List[str], responses: Dict[int, List[str]]):
        M = len(prompts)
        N = len(responses[0])
        all_tokenized = []

        for i, resp in responses.items():
            for r in resp:
                tokenized_pair = self.backbone.tokenize_pair(prompts[i], r)
                all_tokenized.append(tokenized_pair)
        
        encs = self.backbone.tokenizer.pad(
            {"input_ids": all_tokenized},
            return_tensors = "pt"
        )

        device = self.backbone.model.device
        total_pairs = M * N
        features = []

        for ndx in range(0, total_pairs, self.config.backbone_batch):
            batch_end_ndx = min(ndx + self.config.backbone_batch, total_pairs)
            
            batch_enc = {}
            for key, tensor in encs.items():
                batch_enc[key] = tensor[ndx:batch_end_ndx].to(device)

            with torch.no_grad():
                batch_features = self.backbone(**batch_enc) 
                features.append(batch_features)
        
        features = torch.cat(features, dim=0)  # (M*N, 2*D)
        
        # Reshape to (Prompts, Candidates, Features) -> (M, N, 2*D)
        features = features.view(M, N, -1)

        return features

