from typing import Dict, List, Union
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from configs.config import PulseConfig
from transformers.modeling_outputs import BaseModelOutput

class RMbackbone(nn.Module):

    def __init__(self, config: PulseConfig):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(self.config.encoder, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder)
        
        self.source_prefix = "<|source|>"
        self.candidate_prefix = "<|candidate|>"

        self.source_prefix_id = self.tokenizer.convert_tokens_to_ids(self.source_prefix)
        self.cand_prefix_id = self.tokenizer.convert_tokens_to_ids(self.candidate_prefix)

        # add padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # freeze the backbone
        for param in self.model.parameters():
            param.requires_grad = False

        self.eval()

    def tokenize_pair(self, prompt, response):

        source_ids = self.tokenizer.encode(
            self.source_prefix + prompt,
            add_special_tokens = False,
            max_length= self.config.source_max_length,
            truncation=True
        )

        candidate_max_length = self.config.max_length - len(source_ids)

        cand_ids = self.tokenizer.encode(
            self.candidate_prefix + response,
            add_special_tokens = False,
            max_length = candidate_max_length,
            truncation=True
        )
        
        input_ids = source_ids + cand_ids
        
        return input_ids
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        self.model.eval()

        #self.model() is an instance of AutoModel. The call below executes transformers internal forward function.
        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
        )

        return self.postprocess(outputs, input_ids)

    def postprocess(self, outputs: Union[BaseModelOutput, Dict], input_ids: torch.LongTensor):
        encs = outputs.last_hidden_state 
        
        source_idxs = torch.where(input_ids == self.source_prefix_id)

        source_encs = encs[source_idxs[0], source_idxs[1], :]

        cand_idxs = torch.where(input_ids == self.cand_prefix_id)
        
        cand_encs = encs[cand_idxs[0], cand_idxs[1], :]

        source_cand_encs = torch.cat([source_encs, cand_encs], dim=-1)
        
        return source_cand_encs
        
