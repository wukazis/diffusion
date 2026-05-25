import io

import blobfile as bf
import torch


def setup_dist():
    """This project runs in a single process."""


def dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), **kwargs)


def sync_params(_params):
    """Single-process helper kept for API compatibility."""


def get_rank():
    return 0


def get_world_size():
    return 1
