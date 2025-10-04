#!/usr/bin/env python
from nemo.collections import llm
import nemo_run as run
import typer

def merge_lora_with_base_model(peft_path, merged_path):
    return run.Partial(
        llm.peft.merge_lora,
        lora_checkpoint_path=peft_path,
        output_path=merged_path,
    )

def main(
    peft_path: str,
    merged_path: str,
):
    local_executor = run.LocalExecutor()
    run.run(merge_lora_with_base_model(peft_path, merged_path), executor=local_executor)
    print(f"Merged LoRA weights from {peft_path} with base model weights to {merged_path}")

if __name__ == "__main__":
    typer.run(main)
