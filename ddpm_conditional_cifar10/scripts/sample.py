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

from conditional_diffusion import dist_util, logger
from conditional_diffusion.data import CIFAR10_CLASSES
from conditional_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
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
        metadata["class_names"] = list(CIFAR10_CLASSES)
        metadata["sampled_labels"] = labels.tolist()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def build_labels(args, start_index, batch_size, device):
    if not args.class_cond:
        return None

    if args.class_label >= 0:
        if args.class_label >= args.num_classes:
            raise ValueError(f"class_label must be in [0, {args.num_classes - 1}]")
        return th.full((batch_size,), args.class_label, device=device, dtype=th.long)

    labels = th.arange(start_index, start_index + batch_size, device=device, dtype=th.long)
    return labels % args.num_classes


def main():
    args = create_argparser().parse_args()

    if args.class_cond and args.num_classes != len(CIFAR10_CLASSES):
        raise ValueError(f"CIFAR-10 expects num_classes={len(CIFAR10_CLASSES)}")

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating conditional DDPM model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
    sample_start = time.time()
    all_images = []
    all_labels = []
    generated = 0
    while generated < args.num_samples:
        current_batch = min(args.batch_size, args.num_samples - generated)
        labels = build_labels(args, generated, current_batch, dist_util.dev())
        model_kwargs = {"y": labels} if labels is not None else {}
        sample = sample_fn(
            model,
            (current_batch, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()
        all_images.append(sample.cpu().numpy())
        if labels is not None:
            all_labels.append(labels.cpu().numpy())
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
