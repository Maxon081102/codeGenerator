import copy
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, Union, List

from transformers import AutoConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from ..models.Bitnet import BitnetConfig, BitnetForCausalLM
from ..models.GPTBigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM

class ToSmallEmb(nn.Module):
    def __init__(self, emb, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.emb = copy.deepcopy(emb)

    def forward(self, x):
        with torch.no_grad():
            emb = self.emb(x)
        return self.linear(emb)


class ModelWithParent(nn.Module):
    def __init__(self, config: BitnetConfig) -> None:
        super().__init__()
        model_path = '/home/maxim/models/models--bigcode--tiny_starcoder_py/snapshots/8547527bef0bc927268c1653cce6948c5c242dd1'

        model_parent_config = AutoConfig.from_pretrained(
            "bigcode/tiny_starcoder_py", 
            trust_remote_code=True, 
            cache_dir="/home/maxim/models",
            attn_implementation="eager",
        )
        self.parent_model = GPTBigCodeForCausalLM.from_pretrained(
            config=model_parent_config,
            pretrained_model_name_or_path=model_path,
        )
        
        self.model = BitnetForCausalLM(config)
        to_small = ToSmallEmb(self.parent_model.transformer.wte, 768, config.d_model)
        self.model.model.embed_tokens = to_small
        self.parent_model = self.parent_model.half()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_attentions_before_softmax: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            return_attentions_before_softmax=return_attentions_before_softmax,
            )
    
    def train_forward(
        self, 
        x,
        return_attentions=False, 
        calculate_output_2=False,
    ):
        output = self.model(**x, output_attentions=True, return_attentions_before_softmax=return_attentions)
        if calculate_output_2:
            with torch.no_grad():
                output_2 = self.parent_model(**x, output_attentions=True, return_attentions_before_softmax=return_attentions)
                log_p_parent = F.log_softmax(output_2["logits"], dim=-1)
            return output, output_2, log_p_parent
        return output
        