import torch

def test_pytorch():
    x = torch.rand(5, 3)
    print("pytorch is installed with version", torch.__version__)
