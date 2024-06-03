"""Script for a training run."""

import hydra
from hydra.core.hydra_config import HydraConfig

import os
import copy
import torch
import logging

from torch import nn
from datasets import load_dataset, DatasetDict
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from adan import Adan

from generator.train.TrainerWithParent import TrainerWithParent
from generator.train.train_utils.lr_scheduler import BitnetLRScheduler
from generator.models.Llama.modeling_llama import LlamaForCausalLM
from generator.models.Bitnet.modeling_bitnet import BitnetForCausalLM
from generator.train.ModelWithParent import ModelWithParent
from generator.models.Bitnet.configuration_bitnet import (
    BitnetConfig,
    BitnetAttentionConfig,
    BitnetFFNConfig,
)

class ToSmallEmb(nn.Module):
    def __init__(self, emb, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.emb = copy.deepcopy(emb)

    def forward(self, x):
        with torch.no_grad():
            emb = self.emb(x)
        return self.linear(emb)

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def launch(cfg):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    final_model_dir = os.path.join(HydraConfig.get().runtime.output_dir, 'finalModel')
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    transformers.set_seed(cfg.seed)
    ds_train = load_dataset(
        "codeparrot/codeparrot-clean-train",
        split="train",
        cache_dir="/home/maxim/datasets",
    )
    ds_valid = load_dataset(
        "codeparrot/codeparrot-clean-valid",
        split="train",
        cache_dir="/home/maxim/datasets",
    )
    # ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    # ds_valid = load_dataset(
    #     "huggingface-course/codeparrot-ds-valid", split="validation"
    # )

    if cfg.num_token_mult != -1:
        raw_datasets = DatasetDict(
            {
                "train": ds_train.shuffle(seed=0).select(
                    range(cfg.num_token_mult * 100000)
                ),
                "valid": ds_valid.shuffle(seed=0).select(range(2 * 2000)),
            }
        )
    else:
        raw_datasets = DatasetDict(
            {
                "train": ds_train.shuffle(seed=0),
                "valid": ds_valid.shuffle(seed=0).select(range(4 * 2000)),
            }
        )

    context_length = 256
    tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py", trust_remote_code=True, cache_dir="/home/maxim/models")
    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True, cache_dir="/home/maxim/models")
    # model_parent_config = AutoConfig.from_pretrained(
    #     "deepseek-ai/deepseek-coder-1.3b-base", 
    #     trust_remote_code=True, 
    #     cache_dir="/home/maxim/models",
    #     attn_implementation="eager",
    # )
    # model_path = '/home/maxim/models/models--deepseek-ai--deepseek-coder-1.3b-base/snapshots/c919139c3a9b4070729c8b2cca4847ab29ca8d94'
    # parent_model = LlamaForCausalLM.from_pretrained(
    #     config=model_parent_config,
    #     pretrained_model_name_or_path=model_path,
    # )
    # parent_model = AutoModelForCausalLM.from_pretrained(
    #     "deepseek-ai/deepseek-coder-1.3b-base", 
    #     trust_remote_code=True, 
    #     cache_dir="/home/maxim/models",
    #     attn_implementation="eager",
    #     # torch_dtype=torch.float16,
    # )
    vocab_size = 49152

    outputs = tokenizer(
        raw_datasets["train"][:2]["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    print(f"Input IDs length: {len(outputs['input_ids'])}")
    print(f"Input chunk lengths: {(outputs['length'])}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

    def tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    del raw_datasets

    attention_config = BitnetAttentionConfig(
        kv_n_heads=cfg.model.kv_n_heads,
    )

    ffn_config = BitnetFFNConfig(
        ffn_hidden_size=cfg.model.ffn_hidden_size,
    )

    model_config = BitnetConfig(
        attn_implementation="eager",
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        max_seq_len=cfg.model.max_seq_len,
        use_last_bit_linear=cfg.model.use_last_bit_linear,
        vocab_size=vocab_size,
        attn_config=attention_config,
        ffn_config=ffn_config,
    )

    # model = BitnetForCausalLM(model_config).to(torch.float16)
    # to_small = ToSmallEmb(parent_model.model.embed_tokens, 2048, cfg.model.d_model).to(torch.float16)
    # model = BitnetForCausalLM(model_config).to(torch.float32)
    # to_small = ToSmallEmb(parent_model.model.embed_tokens, 2048, cfg.model.d_model).to(torch.float32)
    # to_small = ToSmallEmb(parent_model.model.embed_tokens, 2048, cfg.model.d_model)
    # model.model.embed_tokens = to_small
    # parent_model = parent_model.half()
    model = ModelWithParent(config=model_config)

    model_name = "Bitnet"
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{model_name} size: {model_size/1000**2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="codeparrot-ds",
        per_device_train_batch_size=cfg.train.device_train_batch_size,
        per_device_eval_batch_size=cfg.train.device_eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=50,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        # optim=cfg.train.optim,
        # weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        # lr_scheduler_type=cfg.train.lr_scheduler_type,
        # learning_rate=cfg.train.learning_rate,
        save_steps=500,
        fp16=True,
        # adam_epsilon=cfg.train.adam_epsilon,
        # use_cpu=True,
        max_grad_norm=None,
        # remove_unused_columns=False,
        save_safetensors=False,
    )
    
    # optimizer = Adan(
    #     model.parameters(),
    #     lr=cfg.train.learning_rate,
    #     weight_decay=cfg.train.weight_decay,
    #     betas=(0.98, 0.92, 0.99),
    #     eps=cfg.train.adam_epsilon,
    # )
    
    # max_steps = 10000
    
    # scheduler = BitnetLRScheduler(
    #     optimizer,
    #     num_warmup_steps=args.get_warmup_steps(max_steps),
    #     num_training_steps=max_steps,
    #     second_lr=1e-3,
    #     second_weight_decay=0,
    # )

    # trainer = Trainer(
    trainer = TrainerWithParent(
        config_opt_sch=cfg,
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        # optimizers=(optimizer, scheduler),
    )

    trainer.train()
    trainer.save_model(output_dir=final_model_dir)


if __name__ == "__main__":
    launch()
