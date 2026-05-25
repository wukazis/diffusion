from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

from . import dist_util


def load_paired_data(
    *,
    data_dir,
    batch_size,
    image_size,
    split="train",
    pair_direction="left_to_right",
    deterministic=False,
    num_workers=1,
):
    loader = create_paired_data_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        split=split,
        pair_direction=pair_direction,
        deterministic=deterministic,
        num_workers=num_workers,
        drop_last=not deterministic,
    )
    while True:
        yield from loader


def create_paired_data_loader(
    *,
    data_dir,
    batch_size,
    image_size,
    split="train",
    pair_direction="left_to_right",
    deterministic=False,
    num_workers=1,
    drop_last=True,
):
    split_dir = _resolve_split_dir(data_dir, split)
    all_files = _list_image_files_recursively(split_dir)
    dataset = PairedImageDataset(
        resolution=image_size,
        image_paths=all_files,
        pair_direction=pair_direction,
        shard=dist_util.get_rank(),
        num_shards=dist_util.get_world_size(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def _resolve_split_dir(data_dir, split):
    if not data_dir:
        raise ValueError("unspecified data directory")
    if split:
        split_dir = bf.join(data_dir, split)
        if bf.exists(split_dir) and bf.isdir(split_dir):
            return split_dir
    return data_dir


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp", "webp"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def _preprocess_pil_image(pil_image, resolution):
    while min(*pil_image.size) >= 2 * resolution:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = resolution / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - resolution) // 2
    crop_x = (arr.shape[1] - resolution) // 2
    arr = arr[crop_y : crop_y + resolution, crop_x : crop_x + resolution]
    arr = arr.astype(np.float32) / 127.5 - 1
    return np.transpose(arr, [2, 0, 1])


class PairedImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        pair_direction="left_to_right",
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        if pair_direction not in {"left_to_right", "right_to_left"}:
            raise ValueError(
                "pair_direction must be one of {'left_to_right', 'right_to_left'}"
            )
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.pair_direction = pair_direction

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        pil_image = pil_image.convert("RGB")
        width, height = pil_image.size
        if width < 2:
            raise ValueError(f"paired image must be at least 2 pixels wide: {path}")

        midpoint = width // 2
        left = pil_image.crop((0, 0, midpoint, height))
        right = pil_image.crop((midpoint, 0, width, height))

        if self.pair_direction == "left_to_right":
            cond_image, target_image = left, right
        else:
            cond_image, target_image = right, left

        cond_arr = _preprocess_pil_image(cond_image, self.resolution)
        target_arr = _preprocess_pil_image(target_image, self.resolution)
        return target_arr, {"cond": cond_arr}
