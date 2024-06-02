"""Script for a training run."""

import hydra

import os
import logging

from datasets import load_dataset, DatasetDict
import transformers
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from generator.models.Bitnet.modeling_bitnet import BitnetForCausalLM
from generator.models.Bitnet.configuration_bitnet import (
    BitnetConfig,
    BitnetAttentionConfig,
    BitnetFFNConfig,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def launch(cfg):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
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

    raw_datasets = DatasetDict(
        {
            "train": ds_train.shuffle(seed=0).select(
                range(cfg.num_token_mult * 100000)
            ),
            "valid": ds_valid.shuffle(seed=0).select(range(2 * 2000)),
        }
    )

    context_length = 512
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
    vocab_size = 32000

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

    attention_config = BitnetAttentionConfig(
        kv_n_heads=cfg.model.kv_n_heads,
    )

    ffn_config = BitnetFFNConfig(
        ffn_hidden_size=cfg.model.ffn_hidden_size,
    )

    model_config = BitnetConfig(
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        max_seq_len=cfg.model.max_seq_len,
        use_last_bit_linear=cfg.model.use_last_bit_linear,
        vocab_size=vocab_size,
        attn_config=attention_config,
        ffn_config=ffn_config,
    )

    model = BitnetForCausalLM(model_config)

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
        logging_steps=500,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        optim=cfg.train.optim,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        learning_rate=cfg.train.learning_rate,
        save_steps=10000,
        # fp16=True,
        adam_epsilon=cfg.train.adam_epsilon,
        max_grad_norm=cfg.train.max_grad_norm,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    trainer.train()


if __name__ == "__main__":
    launch()
