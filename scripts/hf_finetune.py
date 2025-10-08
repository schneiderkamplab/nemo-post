#!/usr/bin/env python
from contextlib import suppress
from datasets import load_dataset
from nemo.collections import llm
from nemo.collections import common
from nemo.utils import logging
import nemo_run as run
import os
import typer
from typing import Optional, Dict, Any, List
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import glob

from lightning.pytorch.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from typing import List, Dict, Any, Optional
import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

import torch
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizerBase

class ChatCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_length = seq_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        compute_loss_mask = '{% generation %}' in self.tokenizer.chat_template
        batch = self.tokenizer.apply_chat_template(
            [example["messages"] for example in examples],
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=self.seq_length,
            return_tensors="pt",
            return_assistant_tokens_mask=compute_loss_mask,
            padding=True,
            return_dict=True,
        )
        input_ids = batch["input_ids"][:,:-1]
        labels = batch["input_ids"][:,1:]
        attention_mask = batch["attention_mask"][:,:-1]
        if compute_loss_mask:
            loss_mask = batch["assistant_masks"][:,1:].bool()
            labels = torch.where(loss_mask, labels, torch.full_like(labels, -100))
        result = dict(
            input_ids=input_ids,
            attention_mask=attention_mask[:,:-1],
            labels=labels,
        )
        return result

class HFSFTDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        pretrained_model_name: str,
        seq_length: int,
        micro_batch_size: int,
        tokenizer_name: str,
        num_workers: int = 1,
    ):
        super().__init__()
        self.micro_batch_size = micro_batch_size
        self.dataset_root = dataset_root
        self.model_name = pretrained_model_name
        self.seq_length = seq_length
        self.tokenizer_name = tokenizer_name
        self.num_workers = num_workers

    # def prepare_data(self, stage: Optional[str] = None):
    #     pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        assert hasattr(self.tokenizer, "chat_template")
        self.ds = self._load_dataset(Path(self.dataset_root))
        if not "train" in self.ds or len(self.ds["train"]) == 0:
            raise ValueError("Train split is empty.")
        self.collator = ChatCollator(
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
        )

    def _load_dataset(self, root: Path) -> DatasetDict:
        train_path = root / "training.jsonl"
        val_path = root / "validation.jsonl"
        if train_path.exists():
            files = {"train": str(train_path)}
            if val_path.exists():
                files["validation"] = str(val_path)
            return load_dataset("json", data_files=files)
        raise ValueError(f"No training.jsonl found in {root}")

    def train_dataloader(self):
        return DataLoader(
            self.ds["train"],
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        if "validation" not in self.ds:
            return None
        return DataLoader(
            self.ds["validation"],
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collator,
        )


def local_executor_torchrun(nodes: int, devices: int) -> run.LocalExecutor:
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }
    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)
    return executor

def main(
    nodes: int = 1,
    gpus_per_node: int = 8,
    peft: str = "lora",
    model_name: str = "google/gemma-3-27b-it",
    mbs: int = 1,
    gbs: int = 128,
    lora_r: int = 512,
    lora_alpha: int = 1024,
    seq_length: int = 2048,
    epochs: int = 1,
    data: str = "data",
    tokenizer_name: str = "google/gemma-3-27b-it",
    num_workers: int = 0,
    steps: int = -1,
    strategy: str = 'ddp_find_unused_parameters_true',
):
    name = f"{model_name.replace('/','_')}-{peft}"
    recipe = llm.hf_auto_model_for_causal_lm.finetune_recipe(
        model_name=model_name,
        dir="checkpoints",
        name=name,
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        peft_scheme=peft,
    )
    recipe.data = run.Config(
        HFSFTDataModule,
        dataset_root=data,
        pretrained_model_name=model_name,
        seq_length=seq_length,
        tokenizer_name=tokenizer_name,
        micro_batch_size=mbs,
        num_workers=num_workers,
    )
    if recipe.peft:
        recipe.peft.dim = lora_r
        recipe.peft.alpha = lora_alpha
#        recipe.peft.target_modules = ['*']
    recipe.trainer.strategy = strategy
    recipe.trainer.max_epochs = epochs
    recipe.trainer.max_steps = steps
    recipe.trainer.accumulate_grad_batches = gbs // gpus_per_node // mbs
    assert recipe.trainer.accumulate_grad_batches * gpus_per_node * mbs == gbs
    recipe.trainer.val_check_interval = 100
    recipe.log.ckpt.every_n_train_steps = 1000
    recipe.log.ckpt.train_time_interval = None
    recipe.trainer.log_every_n_steps = 100
    recipe.optim.optimizer_fn.lr = 4e-07
    recipe.optim.lr_scheduler.warmup_steps = 2025 // recipe.trainer.accumulate_grad_batches
    recipe.optim.lr_scheduler.constant_steps = 30000 // recipe.trainer.accumulate_grad_batches
    recipe.optim.lr_scheduler.max_steps = 152025 // recipe.trainer.accumulate_grad_batches
    recipe.optim.lr_scheduler.min_lr = 1e-08
    print(recipe)
    executor = local_executor_torchrun(nodes=nodes, devices=gpus_per_node)
    run.run(recipe, executor=executor, name=name)

if __name__ == "__main__":
    typer.run(main)
