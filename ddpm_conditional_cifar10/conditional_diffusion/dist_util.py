"""
Helpers for distributed training.
"""

import io
import importlib.util
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def _get_mpi():
    if os.environ.get("CONDITIONAL_DIFFUSION_ENABLE_MPI", "0") != "1":
        return None
    if os.environ.get("IMPROVED_DIFFUSION_DISABLE_MPI", "0") == "1":
        return None
    if importlib.util.find_spec("mpi4py") is None:
        return None
    try:
        from mpi4py import MPI
    except Exception:
        return None
    return MPI


def _mpi_available():
    return _get_mpi() is not None


def get_rank():
    mpi = _get_mpi()
    if mpi is not None:
        return mpi.COMM_WORLD.Get_rank()
    return 0


def get_world_size():
    mpi = _get_mpi()
    if mpi is not None:
        return mpi.COMM_WORLD.Get_size()
    return 1


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "nccl"
    world_size = get_world_size()

    if world_size == 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        dist.init_process_group(backend=backend, init_method="env://")
        return

    MPI = _get_mpi()
    comm = MPI.COMM_WORLD

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if get_world_size() == 1:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        return th.load(io.BytesIO(data), **kwargs)

    MPI = _get_mpi()
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
