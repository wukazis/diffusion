"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
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

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    sample_start = time.time()
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    elapsed_tensor = th.tensor(
        [time.time() - sample_start], device=dist_util.dev(), dtype=th.float32
    )
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    elapsed_seconds = float(elapsed_tensor.item())
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
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
            save_run_metadata(
                args, metadata_path, out_path, elapsed_seconds, images_per_second
            )

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_png_samples=64,
        save_grid=True,
        grid_count=64,
        save_metadata=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
