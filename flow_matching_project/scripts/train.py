import argparse
import json
import os

from flow_matching import dist_util, logger
from flow_matching.data import load_data
from flow_matching.flow import FlowMatching
from flow_matching.script_util import (
    add_dict_to_argparser,
    create_model,
    model_kwargs_from_args,
    train_defaults,
)
from flow_matching.train_util import TrainLoop


def save_run_metadata(args, out_dir):
    metadata = vars(args).copy()
    metadata["log_dir"] = out_dir
    with open(os.path.join(out_dir, "train_run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    logger.log(f"device: {dist_util.dev()}")
    logger.log(f"log dir: {logger.get_dir()}")
    save_run_metadata(args, logger.get_dir())
    logger.log("saved run metadata to train_run_metadata.json")

    logger.log("creating FM model...")
    model = create_model(**model_kwargs_from_args(args))
    model.to(dist_util.dev())
    flow = FlowMatching()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=False,
    )
    logger.log(f"data dir: {args.data_dir}")

    logger.log("training...")
    TrainLoop(
        model=model,
        flow=flow,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
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
