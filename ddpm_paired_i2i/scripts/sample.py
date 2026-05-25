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

from paired_diffusion import dist_util, logger
from paired_diffusion.paired_image_datasets import create_paired_data_loader
from paired_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    sample_defaults,
)


def tensor_to_uint8_images(images):
    images = ((images + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return images.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def save_png_samples(arr, out_dir, count):
    os.makedirs(out_dir, exist_ok=True)
    for idx, image in enumerate(arr[:count]):
        Image.fromarray(image).save(os.path.join(out_dir, f"sample_{idx:04d}.png"))


def build_triplet_panels(cond_arr, target_arr, pred_arr):
    return np.concatenate([cond_arr, target_arr, pred_arr], axis=2)


def save_triplet_panels(panels, out_dir, count):
    os.makedirs(out_dir, exist_ok=True)
    for idx, panel in enumerate(panels[:count]):
        Image.fromarray(panel).save(os.path.join(out_dir, f"panel_{idx:04d}.png"))


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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def slice_cond(cond, count):
    return {key: value[:count] for key, value in cond.items()}


def move_cond_to_device(cond):
    return {key: value.to(dist_util.dev()) for key, value in cond.items()}


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating paired DDPM model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    loader = create_paired_data_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        split=args.sample_split,
        pair_direction=args.pair_direction,
        deterministic=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    logger.log("sampling...")
    sample_start = time.time()
    all_images = []
    all_cond = []
    all_target = []
    generated = 0

    for batch, cond in loader:
        if generated >= args.num_samples:
            break

        current_batch = min(batch.shape[0], args.num_samples - generated)
        batch = batch[:current_batch]
        cond = slice_cond(cond, current_batch)

        batch = batch.to(dist_util.dev())
        cond = move_cond_to_device(cond)

        sample = sample_fn(
            model,
            (current_batch, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=cond,
            device=dist_util.dev(),
            progress=args.progress,
        )
        all_images.append(tensor_to_uint8_images(sample))
        all_cond.append(tensor_to_uint8_images(cond["cond"]))
        all_target.append(tensor_to_uint8_images(batch))
        generated += current_batch
        logger.log(f"created {generated} translated samples")

    if generated == 0:
        raise ValueError("no evaluation samples were loaded from the requested split")

    pred_arr = np.concatenate(all_images, axis=0)
    cond_arr = np.concatenate(all_cond, axis=0)
    target_arr = np.concatenate(all_target, axis=0)
    elapsed_seconds = time.time() - sample_start

    shape_str = "x".join([str(x) for x in pred_arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, pred=pred_arr, cond=cond_arr, target=target_arr)

    images_per_second = pred_arr.shape[0] / max(elapsed_seconds, 1e-12)
    logger.log(f"sampling took {elapsed_seconds:.2f}s")
    logger.log(f"throughput: {images_per_second:.2f} images/s")

    panels = build_triplet_panels(cond_arr, target_arr, pred_arr)

    if args.save_png_samples > 0:
        pred_dir = os.path.join(logger.get_dir(), "generated_pngs")
        logger.log(f"saving {min(args.save_png_samples, len(pred_arr))} generated PNGs to {pred_dir}")
        save_png_samples(pred_arr, pred_dir, args.save_png_samples)

        triplet_dir = os.path.join(logger.get_dir(), "triplet_panels")
        logger.log(f"saving {min(args.save_png_samples, len(pred_arr))} triplet panels to {triplet_dir}")
        save_triplet_panels(panels, triplet_dir, args.save_png_samples)

    if args.save_grid:
        pred_grid_path = os.path.join(logger.get_dir(), "sample_grid.png")
        grid_rows = max(1, math.ceil(math.sqrt(args.grid_count)))
        logger.log(f"saving generated sample grid to {pred_grid_path}")
        save_image_grid(pred_arr[: args.grid_count], pred_grid_path, grid_rows)

        panel_grid_path = os.path.join(logger.get_dir(), "triplet_grid.png")
        logger.log(f"saving triplet panel grid to {panel_grid_path}")
        save_image_grid(panels[: args.grid_count], panel_grid_path, grid_rows)

    if args.save_metadata:
        metadata_path = os.path.join(logger.get_dir(), "sample_run_metadata.json")
        logger.log(f"saving run metadata to {metadata_path}")
        save_run_metadata(args, metadata_path, out_path, elapsed_seconds, images_per_second)

    if generated < args.num_samples:
        logger.log(
            f"requested {args.num_samples} samples but only found {generated} items in split {args.sample_split}"
        )

    logger.log("sampling complete")


def create_argparser():
    defaults = sample_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
