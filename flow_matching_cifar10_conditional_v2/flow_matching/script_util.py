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
        use_scale_shift_norm=True,
        class_cond=True,
        num_classes=10,
        class_dropout_prob=0.1,
    )


def train_defaults():
    defaults = dict(
        dataset_name="cifar10",
        data_dir="./data",
        cifar10_train=True,
        cifar10_download=False,
        num_workers=4,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=50000,
        batch_size=128,
        microbatch=32,
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
        num_samples=1000,
        batch_size=128,
        sampling_steps=100,
        sampling_method="heun",
        guidance_scale=1.5,
        model_path="",
        save_png_samples=64,
        save_grid=True,
        grid_count=64,
        save_metadata=True,
        class_label=-1,
        class_name="",
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
    use_scale_shift_norm,
    class_cond,
    num_classes,
    class_dropout_prob,
):
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
        use_scale_shift_norm=use_scale_shift_norm,
        num_classes=num_classes if class_cond else None,
        class_dropout_prob=class_dropout_prob,
    )
