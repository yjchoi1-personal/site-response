#!/bin/bash

module reset

# Load modules
module load gcc/9.1.0
module load python3/3.9
module load cuda/12

# create env
# ---------
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# test env
# --------
echo 'which python -> venv'
which python

echo 'test_pytorch.py -> random tensor'
python test/test_pytorch.py

echo 'test_pytorch_cuda_gpu.py -> True if GPU'
python test/test_pytorch_cuda_gpu.py