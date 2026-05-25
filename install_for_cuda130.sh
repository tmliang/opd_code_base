export MAX_JOBS=32

pip install uv
pip install setuptools wheel ninja cmake
pip install vllm==0.21.0 

# flash-attention
pip install "flash-attn==2.8.3" --no-build-isolation

# deepspeed训练
pip install deepspeed

pip install -r requirements.txt
pip install --no-deps -e .