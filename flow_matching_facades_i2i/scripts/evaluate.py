import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"yes", "true", "t", "y", "1"}:
        return True
    if v.lower() in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_npz", type=str, default="")
    parser.add_argument("--triplet_dir", type=str, default="")
    parser.add_argument("--triplet_glob", type=str, default="panel_*.png")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compute_lpips", type=str2bool, default=False)
    parser.add_argument("--lpips_net", type=str, default="alex")
    parser.add_argument("--save_json", type=str2bool, default=True)
    parser.add_argument("--save_csv", type=str2bool, default=True)
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_from_npz(path):
    data = np.load(path)
    if "pred" not in data or "target" not in data:
        raise ValueError("samples npz must contain 'pred' and 'target' arrays")
    pred = data["pred"]
    target = data["target"]
    names = [f"sample_{idx:04d}" for idx in range(len(pred))]
    return pred, target, names


def load_from_triplet_dir(path, pattern):
    from pathlib import Path

    panel_paths = sorted(Path(path).glob(pattern))
    if not panel_paths:
        raise ValueError(f"no triplet panel images found in {path} with pattern {pattern}")

    preds = []
    targets = []
    names = []
    for panel_path in panel_paths:
        panel = np.array(Image.open(panel_path).convert("RGB"))
        width = panel.shape[1]
        if width % 3 != 0:
            raise ValueError(f"triplet panel width must be divisible by 3: {panel_path}")
        third = width // 3
        target = panel[:, third : 2 * third, :]
        pred = panel[:, 2 * third :, :]
        targets.append(target)
        preds.append(pred)
        names.append(panel_path.stem)
    return np.stack(preds), np.stack(targets), names


def arrays_to_tensor(images):
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("expected images with shape [N, H, W, 3]")
    tensor = torch.from_numpy(images.astype(np.float32) / 255.0)
    return tensor.permute(0, 3, 1, 2).contiguous()


def make_gaussian_kernel(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_per_image(pred, target, window_size=11, sigma=1.5):
    channels = pred.shape[1]
    kernel = make_gaussian_kernel(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


def psnr_per_image(pred, target):
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def batched_metric(metric_fn, pred, target, batch_size, device):
    values = []
    for start in range(0, pred.shape[0], batch_size):
        end = start + batch_size
        pred_batch = pred[start:end].to(device)
        target_batch = target[start:end].to(device)
        batch_values = metric_fn(pred_batch, target_batch)
        values.append(batch_values.detach().cpu())
    return torch.cat(values, dim=0).numpy()


def lpips_per_image(pred, target, batch_size, device, net):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "LPIPS evaluation requires the 'lpips' package. Install it with 'pip install lpips'."
        ) from exc

    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()

    values = []
    for start in range(0, pred.shape[0], batch_size):
        end = start + batch_size
        pred_batch = pred[start:end].to(device) * 2.0 - 1.0
        target_batch = target[start:end].to(device) * 2.0 - 1.0
        with torch.no_grad():
            batch_values = loss_fn(pred_batch, target_batch).view(-1)
        values.append(batch_values.detach().cpu())
    return torch.cat(values, dim=0).numpy()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def save_metrics_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if bool(args.samples_npz) == bool(args.triplet_dir):
        raise ValueError("provide exactly one of --samples_npz or --triplet_dir")

    if args.samples_npz:
        pred_arr, target_arr, names = load_from_npz(args.samples_npz)
        default_output_dir = os.path.dirname(os.path.abspath(args.samples_npz))
    else:
        pred_arr, target_arr, names = load_from_triplet_dir(args.triplet_dir, args.triplet_glob)
        default_output_dir = os.path.abspath(args.triplet_dir)

    if pred_arr.shape != target_arr.shape:
        raise ValueError(
            f"pred and target shapes must match, got {pred_arr.shape} vs {target_arr.shape}"
        )
    if len(pred_arr) == 0:
        raise ValueError("no samples found to evaluate")

    output_dir = os.path.abspath(args.output_dir or default_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = choose_device(args.device)

    pred = arrays_to_tensor(pred_arr)
    target = arrays_to_tensor(target_arr)

    psnr_values = batched_metric(psnr_per_image, pred, target, args.batch_size, device)
    ssim_values = batched_metric(ssim_per_image, pred, target, args.batch_size, device)

    summary = {
        "num_samples": int(len(pred_arr)),
        "input": {
            "samples_npz": os.path.abspath(args.samples_npz) if args.samples_npz else "",
            "triplet_dir": os.path.abspath(args.triplet_dir) if args.triplet_dir else "",
        },
        "metrics": {
            "psnr": summarize(psnr_values),
            "ssim": summarize(ssim_values),
        },
    }

    lpips_values = None
    if args.compute_lpips:
        lpips_values = lpips_per_image(
            pred=pred,
            target=target,
            batch_size=args.batch_size,
            device=device,
            net=args.lpips_net,
        )
        summary["metrics"]["lpips"] = summarize(lpips_values)

    print(f"num_samples: {summary['num_samples']}")
    print(f"PSNR  mean={summary['metrics']['psnr']['mean']:.4f}  std={summary['metrics']['psnr']['std']:.4f}")
    print(f"SSIM  mean={summary['metrics']['ssim']['mean']:.4f}  std={summary['metrics']['ssim']['std']:.4f}")
    if lpips_values is not None:
        print(
            f"LPIPS mean={summary['metrics']['lpips']['mean']:.4f}  std={summary['metrics']['lpips']['std']:.4f}"
        )

    rows = []
    for idx, name in enumerate(names):
        row = {
            "index": idx,
            "name": name,
            "psnr": float(psnr_values[idx]),
            "ssim": float(ssim_values[idx]),
        }
        if lpips_values is not None:
            row["lpips"] = float(lpips_values[idx])
        rows.append(row)

    if args.save_json:
        summary_path = os.path.join(output_dir, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"saved summary to {summary_path}")

    if args.save_csv:
        csv_path = os.path.join(output_dir, "metrics_per_sample.csv")
        save_metrics_csv(csv_path, rows)
        print(f"saved per-sample metrics to {csv_path}")


if __name__ == "__main__":
    main()
