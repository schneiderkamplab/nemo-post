#!/usr/bin/env python
from nemo.collections import llm
import typer

from huggingface_hub import snapshot_download

model_configs = {
    20: llm.GPTOSSConfig20B,
    120: llm.GPTOSSConfig120B,
}

def main(
    model_size: int = 20,
):
    snapshot_download(
        repo_id=f"openai/gpt-oss-{model_size}b",
        local_dir=f"models/gpt-oss-{model_size}b",
        local_dir_use_symlinks=False
    )
    config = model_configs[model_size]()
    model = llm.GPTOSSModel(config=config)
    llm.import_ckpt(
        model=model,
        source=f'hf://models/gpt-oss-{model_size}b',
        output_path=f'models/gpt-oss-{model_size}b-nemo',
        overwrite=True,
    )

if __name__ == "__main__":
    typer.run(main)
