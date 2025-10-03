#!/usr/bin/env python
from nemo.collections import llm
import nemo_run as run
import typer

def configure_export_ckpt(merged_path, export_path):
    return run.Partial(
        llm.export_ckpt,
        path=merged_path,
        target="hf",
        output_path=export_path,
        overwrite=True
    )

def main(
    merged_path: str,
    export_path: str,
):
    local_executor = run.LocalExecutor()
    run.run(configure_export_ckpt(merged_path, export_path), executor=local_executor)
    print(f"Exported merged model from {merged_path} to Hugging Face at {export_path}")

if __name__ == "__main__":
    typer.run(main)
