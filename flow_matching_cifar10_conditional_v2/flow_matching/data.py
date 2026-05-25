from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    dataset_name="cifar10",
    cifar10_train=True,
    cifar10_download=False,
    num_workers=1,
):
    if dataset_name == "cifar10":
        dataset = CIFAR10Dataset(
            resolution=image_size,
            data_dir=data_dir,
            train=cifar10_train,
            class_cond=class_cond,
            download=cifar10_download,
        )
    elif dataset_name == "image_folder":
        if not data_dir:
            raise ValueError("unspecified data directory")
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            class_names = [_infer_class_name(path) for path in all_files]
            sorted_classes = {name: idx for idx, name in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[name] for name in class_names]
        dataset = ImageDataset(image_size, all_files, classes=classes)
    else:
        raise ValueError(f"unsupported dataset_name: {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=not deterministic,
    )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def _infer_class_name(path):
    parent = bf.basename(bf.dirname(path))
    if parent in CIFAR10_CLASSES:
        return parent
    base = bf.basename(path)
    if "_" in base:
        return base.split("_")[0]
    if parent and parent not in (".", "/", "\\"):
        return parent
    return base


def infer_image_folder_class_names(data_dir):
    all_files = _list_image_files_recursively(data_dir)
    class_names = [_infer_class_name(path) for path in all_files]
    return sorted(set(class_names))


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.local_classes = classes

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class CIFAR10Dataset(Dataset):
    def __init__(self, resolution, data_dir, train=True, class_cond=True, download=False):
        super().__init__()
        if not data_dir:
            raise ValueError("unspecified data directory")
        self.resolution = resolution
        self.class_cond = class_cond
        self.dataset = CIFAR10(root=data_dir, train=train, download=download)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pil_image, label = self.dataset[idx]
        pil_image = pil_image.convert("RGB")
        if pil_image.size != (self.resolution, self.resolution):
            pil_image = pil_image.resize((self.resolution, self.resolution), resample=Image.BICUBIC)

        arr = np.array(pil_image, dtype=np.float32) / 127.5 - 1
        out_dict = {}
        if self.class_cond:
            out_dict["y"] = np.array(label, dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
