import argparse

from .model import FMUNetModel


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def model_defaults():
    return dict(
        image_size=32,
        num_channels=128,
        num_res_blocks=2,
        dropout=0.0,
        use_attention=True,
    )


def train_defaults():
    defaults = dict(
        data_dir="",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=5000,
        resume_checkpoint="",
    )
    defaults.update(model_defaults())
    return defaults


def sample_defaults():
    defaults = dict(
        clip_output=True,
        num_samples=10000,
        batch_size=16,
        sampling_steps=100,
        model_path="",
        save_png_samples=64,
        save_grid=True,
        grid_count=64,
        save_metadata=True,
    )
    defaults.update(model_defaults())
    return defaults


def model_kwargs_from_args(args):
    return args_to_dict(args, model_defaults().keys())


def create_model(image_size, num_channels, num_res_blocks, dropout, use_attention):
    if image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")
    attention_levels = (1, 2) if use_attention else ()
    return FMUNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=3,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        channel_mult=channel_mult,
        attention_levels=attention_levels,
    )
