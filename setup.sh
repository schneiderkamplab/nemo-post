#!/bin/bash
# setup of environment
conda create -n gptoss3 python=3.12 -y
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate gptoss3
conda info
export TORCH_CUDA_ARCH_LIST="10.0"

# install CUDA toolkit & CUDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
  && sudo dpkg -i cuda-keyring_1.1-1_all.deb \
  && sudo apt-get update \
  && sudo apt-get -y install cudnn cuda-toolkit-12-8

# install nemo-toolkit
uv pip install transformer-engine[pytorch] \
  nemo-toolkit[all]@git+https://github.com/NVIDIA-NeMo/NeMo.git \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  --pre --upgrade

# install apex
uv pip install apex@git+https://github.com/NVIDIA/apex.git --no-build-isolation \
  --config-settings="--build-option=--cpp_ext" --config-settings="--build-option=--cuda_ext" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

# deacivate bias_activation_fusion (not compatible with all activation functions)
patch $CONDA_PREFIX/lib/python3.12/site-packages/nemo/collections/llm/gpt/model/gpt_oss.py <<'EOF'
84c84
<     bias_activation_fusion: bool = True
---
>     bias_activation_fusion: bool = False
EOF
