import io

import blobfile as bf
import torch as th


def setup_dist():
    """
    Single-process project, so no distributed setup is required.
    """


def dev():
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(_params):
    """
    Single-process project, so there is nothing to synchronize.
    """


def get_rank():
    return 0


def get_world_size():
    return 1
