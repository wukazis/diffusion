import io

import blobfile as bf
import torch

def setup_dist():
    """
    单进程项目，不需要设置分布式环境
    """


def dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), **kwargs)


def sync_params(_params):
    """
    单进程项目无需同步
    """


def get_rank():
    return 0


#只有一个进程
def get_world_size():
    return 1
