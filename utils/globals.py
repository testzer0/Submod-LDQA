import torch
import os

## Roots

ROOT_DIR = "/raid/infolab/adithyabhaskar/submodopt/submission/"
DATA_ROOT = os.path.join(ROOT_DIR, "data")
DATASETS_ROOT = DATA_ROOT
CHECKPT_ROOT = os.path.join(ROOT_DIR, "checkpoints")

## -- Roots

## GPU

GPU_NUMBER = 5

def get_device():
    """
    One needs to call torch.cuda.device('cuda:<GPU>') once before any of this
    """
    if torch.cuda.is_available():
        return torch.device("cuda:{}".format(GPU_NUMBER))
    else:
        return torch.device("cpu")

## -- GPU

if __name__ == '__main__':
    pass