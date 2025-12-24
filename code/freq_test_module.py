import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from torchvision import transforms

import dataloaders.custom_transforms as tr
from dataloaders.dataloader import FundusSegmentation, MNMSSegmentation, ProstateSegmentation


def extract_amp_spectrum(img_np: np.ndarray) -> np.ndarray:
    """Compute amplitude spectrum for a CxHxW numpy image."""

    fft = np.fft.fft2(img_np, axes=(-2, -1))
    amp_np, _ = np.abs(fft), np.angle(fft)
    return amp_np


def low_freq_mutate_np(amp_src: np.ndarray, amp_trg: np.ndarray, L: float = 0.1, degree: float = 1) -> np.ndarray:
    """Replace the centered low-frequency patch of ``amp_src`` with ``amp_trg``.

    This mirrors the augmentation used in training: the size of the swapped block
    is controlled by ``L`` and the interpolation ratio by ``degree``.
    """

    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    ratio = random.uniform(0, degree)

    a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * (1 - ratio) + a_trg[:, h1:h2, w1:w2] * ratio
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def source_to_target_freq(src_img: np.ndarray, amp_trg: np.ndarray, L: float = 0.1, degree: float = 1) -> np.ndarray:
    """Swap low-frequency amplitude of ``src_img`` with ``amp_trg`` and return spatial image."""

    fft_src_np = np.fft.fft2(src_img, axes=(-2, -1))
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L, degree=degree)
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)
    return src_in_trg


def _get_dataset_config(dataset: str) -> Dict:
    dataset = dataset.lower()
    if dataset == "fundus":
        return {
            "cls": FundusSegmentation,
            "patch_size": 256,
            "fillcolor": 255,
            "min_v": 0.5,
            "max_v": 1.5,
            "num_channels": 3,
            "base_dir": "../../data/Fundus",
        }
    if dataset == "prostate":
        return {
            "cls": ProstateSegmentation,
            "patch_size": 384,
            "fillcolor": 255,
            "min_v": 0.1,
            "max_v": 2.0,
            "num_channels": 1,
            "base_dir": "../../data/ProstateSlice",
        }
    if dataset == "mnms":
        return {
            "cls": MNMSSegmentation,
            "patch_size": 288,
            "fillcolor": 0,
            "min_v": 0.1,
            "max_v": 2.0,
            "num_channels": 1,
            "base_dir": "../../data/MNMS/mnms",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def _build_transforms(patch_size: int, fillcolor: int, min_v: float, max_v: float, num_channels: int) -> Tuple[transforms.Compose, transforms.Compose]:
    weak = transforms.Compose(
        [
            tr.RandomScaleCrop(patch_size),
            tr.RandomScaleRotate(fillcolor=fillcolor),
            tr.RandomHorizontalFlip(),
            tr.elastic_transform(),
        ]
    )

    normal_to_tensor = transforms.Compose(
        [
            tr.Normalize_tf(),
            tr.ToTensor(),
        ]
    )

    # Strong augmentation is defined to mirror training but not applied in this test module.
    _ = transforms.Compose(
        [
            tr.Brightness(min_v, max_v),
            tr.Contrast(min_v, max_v),
            tr.GaussianBlur(kernel_size=int(0.1 * patch_size), num_channels=num_channels),
        ]
    )
    return weak, normal_to_tensor


def _load_sample(dataset_cls, base_dir: str, domain: int, weak, normal_to_tensor, index: int):
    dataset = dataset_cls(
        base_dir=base_dir,
        phase="train",
        splitid=-1,
        domain=[domain],
        weak_transform=weak,
        normal_toTensor=normal_to_tensor,
    )
    if index >= len(dataset):
        raise IndexError(f"Requested index {index} out of range for domain {domain} (size {len(dataset)}).")
    return dataset[index]


def _denorm_to_uint8(img_tensor: np.ndarray) -> np.ndarray:
    img = ((img_tensor + 1.0) * 127.5).clip(0, 255)
    img = img.astype(np.uint8)
    if img.shape[0] == 1:
        img = img.squeeze(0)
    else:
        img = np.transpose(img, (1, 2, 0))
    return img


def _save_image(array: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if array.ndim == 2:
        img = Image.fromarray(array, mode="L")
    else:
        img = Image.fromarray(array)
    img.save(path)


def run_frequency_transfer(args):
    cfg = _get_dataset_config(args.dataset)
    weak, normal_to_tensor = _build_transforms(
        cfg["patch_size"], cfg["fillcolor"], cfg["min_v"], cfg["max_v"], cfg["num_channels"]
    )
    base_dir = args.base_dir or cfg["base_dir"]

    src_sample = _load_sample(cfg["cls"], base_dir, args.source_domain, weak, normal_to_tensor, args.source_index)
    tgt_sample = _load_sample(cfg["cls"], base_dir, args.target_domain, weak, normal_to_tensor, args.target_index)

    src_img = src_sample["image"].numpy()
    tgt_img = tgt_sample["image"].numpy()

    amp_trg = extract_amp_spectrum((tgt_img + 1) * 127.5)
    freq_img = source_to_target_freq((src_img + 1) * 127.5, amp_trg, L=args.L, degree=args.degree)
    freq_img = np.clip(freq_img, 0, 255).astype(np.float32)
    freq_tensor = freq_img / 127.5 - 1

    src_vis = _denorm_to_uint8(src_img)
    tgt_vis = _denorm_to_uint8(tgt_img)
    freq_vis = _denorm_to_uint8(freq_tensor)

    src_path = os.path.join(args.output_dir, f"source_d{args.source_domain}_idx{args.source_index}.png")
    tgt_path = os.path.join(args.output_dir, f"target_d{args.target_domain}_idx{args.target_index}.png")
    freq_path = os.path.join(args.output_dir, f"freq_d{args.source_domain}_to_d{args.target_domain}.png")

    _save_image(src_vis, src_path)
    _save_image(tgt_vis, tgt_path)
    _save_image(freq_vis, freq_path)

    print(f"Saved source image to: {src_path}")
    print(f"Saved target image to: {tgt_path}")
    print(f"Saved frequency-mixed image to: {freq_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test frequency-domain mixing between two domains.")
    parser.add_argument("--dataset", type=str, default="prostate", choices=["fundus", "prostate", "mnms"], help="Dataset name (matches training setup).")
    parser.add_argument("--base_dir", type=str, default="", help="Override dataset root; defaults follow training scripts.")
    parser.add_argument("--source_domain", type=int, default=1, help="Domain id for the source image.")
    parser.add_argument("--target_domain", type=int, default=2, help="Domain id for the target image.")
    parser.add_argument("--source_index", type=int, default=0, help="Index of the source image within its domain.")
    parser.add_argument("--target_index", type=int, default=0, help="Index of the target image within its domain.")
    parser.add_argument("--L", type=float, default=0.01, help="Low-frequency ratio used during training (matches args.LB).")
    parser.add_argument("--degree", type=float, default=1.0, help="Upper bound for the interpolation ratio used in low-frequency swapping.")
    parser.add_argument("--output_dir", type=str, default="./freq_outputs", help="Directory to save visualized results.")
    return parser.parse_args()


if __name__ == "__main__":
    run_frequency_transfer(parse_args())
