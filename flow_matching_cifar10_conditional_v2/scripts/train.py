import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flow_matching import dist_util, logger
from flow_matching.data import CIFAR10_CLASSES, infer_image_folder_class_names, load_data
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
    if args.dataset_name == "cifar10":
        metadata["class_names"] = list(CIFAR10_CLASSES)
    elif args.dataset_name == "image_folder" and args.class_cond:
        metadata["class_names"] = infer_image_folder_class_names(args.data_dir)
    with open(os.path.join(out_dir, "train_run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main():
    args = create_argparser().parse_args()

    if args.dataset_name == "cifar10" and args.num_classes != len(CIFAR10_CLASSES):
        raise ValueError(f"CIFAR-10 expects num_classes={len(CIFAR10_CLASSES)}")

    dist_util.setup_dist()
    logger.configure()
    logger.log(f"device: {dist_util.dev()}")
    logger.log(f"log dir: {logger.get_dir()}")
    save_run_metadata(args, logger.get_dir())
    logger.log("saved run metadata to train_run_metadata.json")

    logger.log("creating conditional FM model...")
    model = create_model(**model_kwargs_from_args(args))
    model.to(dist_util.dev())
    flow = FlowMatching()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        dataset_name=args.dataset_name,
        cifar10_train=args.cifar10_train,
        cifar10_download=args.cifar10_download,
        num_workers=args.num_workers,
    )
    logger.log(f"dataset: {args.dataset_name}")
    logger.log(f"data dir: {args.data_dir}")
    logger.log(f"class conditional: {args.class_cond}")
    logger.log(f"num classes: {args.num_classes}")
    logger.log(f"class dropout prob: {args.class_dropout_prob}")

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
