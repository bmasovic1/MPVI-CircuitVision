"""Konverzija DeepPCB -> YOLO format (train/val/test + dataset.yaml).

Primjer:  python -m src.data.to_yolo --root data/DeepPCB --out data/yolo
Validacijski skup se izdvaja iz trainval PO GRUPAMA da nema curenja podataka.
"""
import argparse
import shutil
from pathlib import Path

from .deeppcb import NAMES, read_ann, read_split, split_by_group

IMG_SIZE = 640


def write_split(items, out, split):
    img_dir = Path(out) / "images" / split
    lbl_dir = Path(out) / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        name = it["test"].stem  # npr. 20085000_test
        shutil.copy(it["test"], img_dir / f"{name}.jpg")
        lines = []
        for x1, y1, x2, y2, t in read_ann(it["ann"]):
            cx, cy = (x1 + x2) / 2 / IMG_SIZE, (y1 + y2) / 2 / IMG_SIZE
            w, h = (x2 - x1) / IMG_SIZE, (y2 - y1) / IMG_SIZE
            lines.append(f"{t - 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/DeepPCB")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    train, val = split_by_group(read_split("trainval", a.root), a.val_fraction, a.seed)
    test = read_split("test", a.root)
    for items, split in [(train, "train"), (val, "val"), (test, "test")]:
        write_split(items, a.out, split)
        print(f"{split}: {len(items)} slika")

    out = Path(a.out).resolve()
    names = "\n".join(f"  {k - 1}: {v}" for k, v in NAMES.items())
    (out / "dataset.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"
    )
    print("Zapisano:", out / "dataset.yaml")


if __name__ == "__main__":
    main()
