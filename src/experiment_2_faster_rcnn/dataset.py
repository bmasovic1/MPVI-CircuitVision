"""DeepPCB u obliku koji očekuje torchvision Faster R-CNN.

Svaki uzorak je par (image, target):
- image:  float tensor (3, 640, 640), vrijednosti 0-1. Slike su sive (binarne), pa se isti kanal
          ponavlja tri puta jer pretrained model (COCO) očekuje RGB ulaz.
- target: dict s ključevima
          boxes    float tensor (N, 4), pikseli originalne slike, (x1, y1, x2, y2)
          labels   int64 tensor (N,), klase 1-6 kao u DeepPCB-u (0 je pozadina u torchvisionu)
          image_id string, npr. '20085000' (isti ID koji traži docs/FORMAT.md)

Split je isti kao za sve ostale eksperimente (src/evaluation/metrics.get_split_items, seed 42).
"""
import numpy as np
import torch
from PIL import Image

from src.data.deeppcb import DEFAULT_ROOT, NAMES, read_ann
from src.evaluation.metrics import get_split_items, image_id


def load_gray(path):
    """Učitava sliku kao sivu (neke DeepPCB slike su u RGB modu, pa se normalizuje na 'L')."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


class DeepPCBDetection(torch.utils.data.Dataset):
    """Testne slike DeepPCB-a s anotacijama, za split 'train', 'val' ili 'test'.

    transforms: opcionalna funkcija (image, target) -> (image, target), npr. augmentacija.
    """

    def __init__(self, split="train", root=DEFAULT_ROOT, transforms=None):
        self.items = get_split_items(split, root)
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        gray = torch.from_numpy(load_gray(item["test"]))
        image = gray.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)

        rows = read_ann(item["ann"])
        boxes = torch.tensor([r[:4] for r in rows], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([r[4] for r in rows], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id(item)}

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    """Slike u batchu se ne slažu u jedan tensor (različit broj boxova), nego ostaju liste."""
    return tuple(zip(*batch))


def show_sample(image, target, ax=None):
    """Crta uzorak iz DeepPCBDetection (za provjeru da boxovi stoje na defektima)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image[0].numpy(), cmap="gray")
    for (x1, y1, x2, y2), lab in zip(target["boxes"].tolist(), target["labels"].tolist()):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
        ax.text(x1, y1 - 3, NAMES[lab], color="red", fontsize=9, fontweight="bold")
    ax.set_title(f"{target['image_id']}  ({len(target['labels'])} defekata)")
    ax.axis("off")
    return ax
