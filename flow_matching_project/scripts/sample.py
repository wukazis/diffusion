import argparse
import json
import math
import os
import time

import numpy as np
import torch as th
from PIL import Image

from flow_matching import dist_util, logger
from flow_matching.flow import FlowMatching
from flow_matching.script_util import (
    add_dict_to_argparser,
    create_model,
    model_kwargs_from_args,
    sample_defaults,
)


def save_png_samples(arr, out_dir, count):
    os.makedirs(out_dir, exist_ok=True)
    for idx, image in enumerate(arr[:count]):
        Image.fromarray(image).save(os.path.join(out_dir, f"sample_{idx:04d}.png"))


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


def save_run_metadata(args, out_path, sample_path, elapsed_seconds, images_per_second):
    metadata = vars(args).copy()
    metadata.update(
        {
            "sample_path": sample_path,
            "elapsed_seconds": elapsed_seconds,
            "images_per_second": images_per_second,
        }
    )
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating FM model...")
    model = create_model(**model_kwargs_from_args(args))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()
    flow = FlowMatching()

    logger.log("sampling...")
    sample_start = time.time()
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        sample = flow.sample_euler(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            device=dist_util.dev(),
            steps=args.sampling_steps,
            clip_output=args.clip_output,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()
        all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    elapsed_seconds = time.time() - sample_start

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr)

    images_per_second = arr.shape[0] / max(elapsed_seconds, 1e-12)
    logger.log(f"sampling took {elapsed_seconds:.2f}s")
    logger.log(f"throughput: {images_per_second:.2f} images/s")

    if args.save_png_samples > 0:
        png_dir = os.path.join(logger.get_dir(), "png_samples")
        logger.log(f"saving {min(args.save_png_samples, len(arr))} PNG samples to {png_dir}")
        save_png_samples(arr, png_dir, args.save_png_samples)
    if args.save_grid:
        grid_path = os.path.join(logger.get_dir(), "sample_grid.png")
        grid_rows = max(1, math.ceil(math.sqrt(args.grid_count)))
        logger.log(f"saving sample grid to {grid_path}")
        save_image_grid(arr[: args.grid_count], grid_path, grid_rows)
    if args.save_metadata:
        metadata_path = os.path.join(logger.get_dir(), "sample_run_metadata.json")
        logger.log(f"saving run metadata to {metadata_path}")
        save_run_metadata(args, metadata_path, out_path, elapsed_seconds, images_per_second)

    logger.log("sampling complete")


def create_argparser():
    defaults = sample_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
