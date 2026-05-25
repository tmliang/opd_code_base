set -e

if [ ! -d "/usr/local/cuda-12.8" ]; then
    echo "ERROR: CUDA 12.8 not found at /usr/local/cuda-12.8" >&2
    exit 1
fi

if ! /usr/local/cuda-12.8/bin/nvcc --version | grep -q "release 12.8"; then
    echo "ERROR: nvcc at /usr/local/cuda-12.8 is not version 12.8" >&2
    /usr/local/cuda-12.8/bin/nvcc --version >&2
    exit 1
fi

export MAX_JOBS=32
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install uv
pip install setuptools wheel ninja cmake

git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build/cuda.txt
uv pip install --no-build-isolation -e .
VLLM_USE_PRECOMPILED=0 uv pip install vllm==0.21.0 --no-binary=vllm --torch-backend=auto
cd ..
rm -rf vllm

# flash-attention
pip install "flash-attn==2.8.3" --no-build-isolation

pip install deepspeed

# verl
pip install -r requirements.txt
pip install --no-deps -e .