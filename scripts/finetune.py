#!/usr/bin/env python
from datasets import load_dataset
from nemo.collections import llm
from nemo.collections import common
from nemo.utils import logging
import nemo_run as run
import os
import typer

recipes_map = {
    20: llm.recipes.gpt_oss_20b.finetune_recipe,
    120: llm.recipes.gpt_oss_120b.finetune_recipe,
}

def configure_recipe(nodes, gpus_per_node, name, peft, model_size, ep, tp, mbs, lora_r, lora_alpha):
    recipe = recipes_map[model_size](
        name=name,
        resume_path=os.path.abspath(f"models/gpt-oss-{model_size}b-nemo"),
        dir=f"checkpoints",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        peft_scheme=peft,
        packed_sequence=False,
    )
    recipe.trainer.strategy.ckpt_load_strictness = False
    recipe.trainer.strategy.tensor_model_parallel_size = tp
    recipe.trainer.strategy.pipeline_model_parallel_size = 1
    recipe.trainer.strategy.expert_model_parallel_size = ep
    recipe.trainer.strategy.expert_tensor_parallel_size = 1
    recipe.trainer.strategy.sequence_parallel = tp > 1
    recipe.model.config.tensor_model_parallel_size = tp
    recipe.model.config.pipeline_model_parallel_size = 1
    recipe.model.config.expert_model_parallel_size = ep
    recipe.model.config.expert_tensor_parallel_size = 1
    recipe.model.config.sequence_parallel = tp > 1
    recipe.model.config.bias_activation_fusion = False
    recipe.trainer.max_steps = 7264
    if recipe.peft:
        recipe.peft.dim = lora_r
        recipe.peft.alpha = lora_alpha
    recipe.trainer.val_check_interval = 200
    recipe.log.ckpt.every_n_train_steps = 200
    recipe.trainer.log_every_n_steps = 1
    recipe.data = run.Config(
        llm.gpt.data.chat.ChatDataModule,
        dataset_root = "data",
        tokenizer = run.Config(
            common.tokenizers.huggingface.auto_tokenizer.AutoTokenizer,
            pretrained_model_name=f"models/gpt-oss-{model_size}b",
            use_fast=True,
        ),
        global_batch_size = 128,
        micro_batch_size = mbs,
        use_hf_tokenizer_chat_template = True,
    )
    print(recipe)
    return recipe

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
    peft: str = "none",
    model_size: int = 20,
    ep: int = 8,
    tp: int = 1,
    mbs: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
):
    name = f"gptoss_{model_size}b_{peft}"
    recipe = configure_recipe(nodes=nodes, gpus_per_node=gpus_per_node, name=name, peft=peft, model_size=model_size, ep=ep, tp=tp, mbs=mbs, lora_r=lora_r, lora_alpha=lora_alpha)
    executor = local_executor_torchrun(nodes=nodes, devices=gpus_per_node)
    run.run(recipe, executor=executor, name=name)

if __name__ == "__main__":
    typer.run(main)
