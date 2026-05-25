import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch as th
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flow_matching import dist_util, logger
from flow_matching.data import CIFAR10_CLASSES
from flow_matching.flow import FlowMatching
from flow_matching.script_util import (
    add_dict_to_argparser,
    create_model,
    model_kwargs_from_args,
    sample_defaults,
)


def save_png_samples(arr, out_dir, count, labels=None):
    os.makedirs(out_dir, exist_ok=True)
    for idx, image in enumerate(arr[:count]):
        suffix = ""
        if labels is not None:
            suffix = f"_class_{int(labels[idx])}"
        Image.fromarray(image).save(os.path.join(out_dir, f"sample_{idx:04d}{suffix}.png"))


def save_image_grid(arr, out_path, rows):
    if len(arr) == 0:
        return
    rows = max(1, rows)
    count = min(len(arr), rows * rows)
    h, w, c = arr[0].shape
    grid = np.zeros((rows * h, rows * w, c), dtype=np.uint8)
    for idx, image in enumerate(arr[:count]):
        row = idx // rows
        col = idx % rows
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    Image.fromarray(grid).save(out_path)


def save_run_metadata(args, out_path, sample_path, elapsed_seconds, images_per_second, labels):
    metadata = vars(args).copy()
    metadata.update(
        {
            "sample_path": sample_path,
            "elapsed_seconds": elapsed_seconds,
            "images_per_second": images_per_second,
        }
    )
    if labels is not None:
        metadata["class_names"] = resolve_class_names(args)
        metadata["sampled_labels"] = labels.tolist()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_training_metadata(model_path):
    metadata_path = os.path.join(os.path.dirname(model_path), "train_run_metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_class_names(args):
    metadata = load_training_metadata(args.model_path)
    class_names = metadata.get("class_names")
    if class_names:
        return class_names
    return list(CIFAR10_CLASSES)


def resolve_class_label(args):
    if not args.class_name:
        return args.class_label
    class_names = resolve_class_names(args)
    if args.class_name not in class_names:
        raise ValueError(f"class_name must be one of: {class_names}")
    return class_names.index(args.class_name)


def build_model_kwargs(args, start_index, batch_size, device):
    if not args.class_cond:
        return {}

    class_label = resolve_class_label(args)
    if class_label >= 0:
        if class_label >= args.num_classes:
            raise ValueError(f"class_label must be in [0, {args.num_classes - 1}]")
        labels = th.full((batch_size,), class_label, device=device, dtype=th.long)
    else:
        labels = th.arange(start_index, start_index + batch_size, device=device, dtype=th.long)
        labels = labels % args.num_classes

    return {"y": labels}


def main():
    args = create_argparser().parse_args()

    if args.class_cond and args.num_classes != len(CIFAR10_CLASSES):
        raise ValueError(f"CIFAR-10 expects num_classes={len(CIFAR10_CLASSES)}")

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating conditional FM model...")
    model = create_model(**model_kwargs_from_args(args))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()
    flow = FlowMatching()

    logger.log("sampling...")
    sample_start = time.time()
    all_images = []
    all_labels = []
    class_names = resolve_class_names(args)
    logger.log(f"class names: {class_names}")
    generated = 0
    while generated < args.num_samples:
        current_batch = min(args.batch_size, args.num_samples - generated)
        model_kwargs = build_model_kwargs(args, generated, current_batch, dist_util.dev())
        if args.sampling_method == "heun":
            sample = flow.sample_heun(
                model,
                (current_batch, 3, args.image_size, args.image_size),
                device=dist_util.dev(),
                steps=args.sampling_steps,
                model_kwargs=model_kwargs,
                clip_output=args.clip_output,
                guidance_scale=args.guidance_scale,
            )
        else:
            sample = flow.sample_euler(
                model,
                (current_batch, 3, args.image_size, args.image_size),
                device=dist_util.dev(),
                steps=args.sampling_steps,
                model_kwargs=model_kwargs,
                clip_output=args.clip_output,
                guidance_scale=args.guidance_scale,
            )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()
        all_images.append(sample.cpu().numpy())
        if model_kwargs:
            all_labels.append(model_kwargs["y"].cpu().numpy())
        generated += current_batch
        logger.log(f"created {generated} samples")

    arr = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else None
    elapsed_seconds = time.time() - sample_start

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    if labels is None:
        np.savez(out_path, arr=arr)
    else:
        np.savez(out_path, arr=arr, labels=labels)

    images_per_second = arr.shape[0] / max(elapsed_seconds, 1e-12)
    logger.log(f"sampling took {elapsed_seconds:.2f}s")
    logger.log(f"throughput: {images_per_second:.2f} images/s")

    if args.save_png_samples > 0:
        png_dir = os.path.join(logger.get_dir(), "png_samples")
        logger.log(f"saving {min(args.save_png_samples, len(arr))} PNG samples to {png_dir}")
        save_png_samples(arr, png_dir, args.save_png_samples, labels=labels)
    if args.save_grid:
        grid_path = os.path.join(logger.get_dir(), "sample_grid.png")
        grid_rows = max(1, math.ceil(math.sqrt(args.grid_count)))
        logger.log(f"saving sample grid to {grid_path}")
        save_image_grid(arr[: args.grid_count], grid_path, grid_rows)
    if args.save_metadata:
        metadata_path = os.path.join(logger.get_dir(), "sample_run_metadata.json")
        logger.log(f"saving run metadata to {metadata_path}")
        save_run_metadata(args, metadata_path, out_path, elapsed_seconds, images_per_second, labels)

    logger.log("sampling complete")


def create_argparser():
    defaults = sample_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
