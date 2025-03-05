import torch
import pytest

def test_pytorch_cuda_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        pytest.warns(UserWarning, match="CUDA not available")
    else:
        print("CUDA is available")
        
    print(device)
