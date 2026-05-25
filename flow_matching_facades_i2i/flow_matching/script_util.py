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
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        dropout=0.0,
        use_attention=True,
        cond_channels=3,
    )


def train_defaults():
    defaults = dict(
        data_dir="",
        train_split="train",
        pair_direction="left_to_right",
        num_workers=4,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=20000,
        batch_size=4,
        microbatch=1,
        ema_rate="0.9999",
        log_interval=50,
        save_interval=2000,
        resume_checkpoint="",
    )
    defaults.update(model_defaults())
    return defaults


def sample_defaults():
    defaults = dict(
        data_dir="",
        sample_split="val",
        pair_direction="left_to_right",
        num_workers=1,
        clip_output=True,
        num_samples=16,
        batch_size=4,
        sampling_steps=100,
        model_path="",
        save_png_samples=16,
        save_grid=True,
        grid_count=16,
        save_metadata=True,
    )
    defaults.update(model_defaults())
    return defaults


def model_kwargs_from_args(args):
    return args_to_dict(args, model_defaults().keys())


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    dropout,
    use_attention,
    cond_channels,
):
    channel_mult, attention_levels = model_arch_defaults(image_size, use_attention)
    return FMUNetModel(
        in_channels=3,
        cond_channels=cond_channels,
        model_channels=num_channels,
        out_channels=3,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        channel_mult=channel_mult,
        attention_levels=attention_levels,
    )


def model_arch_defaults(image_size, use_attention):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
        attention_levels = (3, 4, 5)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
        attention_levels = (2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
        attention_levels = (1, 2)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
        attention_levels = (1, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")
    if not use_attention:
        attention_levels = ()
    return channel_mult, attention_levels
