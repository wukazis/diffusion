import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paired_diffusion import dist_util, logger
from paired_diffusion.paired_image_datasets import load_paired_data
from paired_diffusion.resample import create_named_schedule_sampler
from paired_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    train_defaults,
)
from paired_diffusion.train_util import TrainLoop


def save_run_metadata(args, out_dir):
    metadata = vars(args).copy()
    metadata["log_dir"] = out_dir
    with open(os.path.join(out_dir, "train_run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    if dist_util.dev() is not None:
        logger.log(f"device: {dist_util.dev()}")
    logger.log(f"log dir: {logger.get_dir()}")
    if logger.get_dir() and dist_util.get_rank() == 0:
        save_run_metadata(args, logger.get_dir())
        logger.log("saved run metadata to train_run_metadata.json")

    logger.log("creating paired DDPM model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating paired data loader...")
    data = load_paired_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        split=args.train_split,
        pair_direction=args.pair_direction,
        num_workers=args.num_workers,
    )
    logger.log(f"data dir: {args.data_dir}")
    logger.log(f"train split: {args.train_split}")
    logger.log(f"pair direction: {args.pair_direction}")
    logger.log(f"schedule sampler: {args.schedule_sampler}")

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = train_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
