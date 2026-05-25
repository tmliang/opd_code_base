#!/bin/bash
#
# Install script aligned with the ms-swift recommended combination:
#   transformers==5.2.*  +  vllm>=0.17.0 (ships with torch 2.10)  +  flash-attn==2.8.3
# Notes:
#   * transformers 5.2 is required: 5.3 breaks video training, and 5.x is
#     incompatible with older vllm. See
#       https://github.com/modelscope/ms-swift/issues/8254
#       https://github.com/modelscope/ms-swift/issues/8362
#   * Python 3.12 is required for flash-linear-attention:
#       https://github.com/fla-org/flash-linear-attention/issues/121

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}
USE_DEEPSPEED=${USE_DEEPSPEED:-1}

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    pip install "sglang[all]==0.5.2" --no-cache-dir && pip install torch-memory-saver --no-cache-dir
fi
# vllm>=0.17.0 pulls in torch 2.10; do NOT pin torch separately or it will conflict.
pip install -U --no-cache-dir "vllm>=0.17.0"
# For RL training vLLM bundles an older transformers; force the ms-swift combo back on top.
pip install -U "transformers==5.2.*"

echo "2. install basic packages"
pip install -U \
    "transformers[hf_xet]==5.2.*" "qwen_vl_utils>=0.0.14" peft liger-kernel \
    ms-swift accelerate datasets hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc wandb dill pybind11 mathruler \
    pytest py-spy pre-commit ruff tensorboard

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention / FlashInfer / FLA / causal-conv1d"
# flash-attn 2.8.3 has no prebuilt wheels for torch 2.10 yet, build from source.
pip install "flash-attn==2.8.3" --no-build-isolation
# flash-linear-attention (training-speed fix: https://github.com/fla-org/flash-linear-attention/issues/758)
pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation
# causal-conv1d (Mamba/RWKV-style ops)
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation

pip install --no-cache-dir flashinfer-python==0.3.1


if [ $USE_DEEPSPEED -eq 1 ]; then
    echo "3b. install DeepSpeed"
    pip install deepspeed
fi


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    pip install "onnxscript==0.3.1"
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
fi


echo "5. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    pip install nvidia-cudnn-cu12==9.10.2.21
fi

echo "Successfully installed all packages"
