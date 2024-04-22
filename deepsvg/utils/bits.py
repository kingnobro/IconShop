import torch

# ref: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits

def int2bit(x, bits=8):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bit2int(x, bits=8):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return torch.sum(mask * x, -1)