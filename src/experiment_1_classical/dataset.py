"""Učitavanje i provjera DeepPCB parova slika (test/template) i anotacija za Eksperiment 1.

Ovaj modul ne definiše vlastitu podjelu podataka: koristi istu podjelu kao `src/evaluation` i
`src/data/to_yolo.py` (`split_by_group`, seed 42), pa svi eksperimenti dijele isti train/val/test.

Primjer:
    cfg = ExperimentConfig()
    train = get_items("train", cfg); val = get_items("val", cfg)
    ensure_group_disjoint(train, val)               # nema iste ploče u treningu i validaciji
    for pair in iter_pairs(val):                    # slike se učitavaju jedna po jedna (štedi memoriju)
        pair.test, pair.temp, pair.boxes, pair.cls

Provjera dataseta (bez treniranja):  python -m src.experiment_1_classical.dataset --split val
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image

from src.data.deeppcb import NAMES, read_ann
from src.evaluation.metrics import get_split_items, image_id

from .config import DEFECT_CLASSES, ExperimentConfig

IMG_SIZE = 640
SPLIT_NAMES = ("train", "val", "test")
Item = dict[str, Any]  # zapis iz src.data.deeppcb: test, temp, ann (Path) i group (str)


@dataclass(frozen=True)
class PcbPair:
    """Jedan par slika s ground truthom. Slike su sivi tonovi uint8 (640, 640); boxovi x1, y1, x2, y2 u pikselima."""
    image_id: str
    group: str
    test: np.ndarray
    temp: np.ndarray
    boxes: np.ndarray  # (N, 4) float
    cls: np.ndarray  # (N,) int, klase 1-6


# ----------------------------------------------------------------------------- jedna slika / anotacija

def load_gray(path: str | Path) -> np.ndarray:
    """Učitava sliku kao sive tonove uint8 (640, 640).

    Dio slika u DeepPCB-u je u RGB modu (62 od 3000), pa se uvijek radi `convert("L")` da oblici nizova budu isti.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"slika ne postoji: {p}")
    try:
        with Image.open(p) as im:
            arr = np.array(im.convert("L"), dtype=np.uint8)
    except OSError as e:  # uključuje PIL.UnidentifiedImageError (oštećen ili neispravan fajl)
        raise ValueError(f"slika se ne može pročitati: {p} ({e})") from e
    if arr.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(f"očekivana veličina {IMG_SIZE}x{IMG_SIZE}, a {p} je {arr.shape[1]}x{arr.shape[0]}")
    return arr


def load_annotations(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Čita anotacije (x1 y1 x2 y2 tip) i provjerava ih. Vraća (boxes Nx4 float, cls N int).

    Prazan fajl je dozvoljen (slika bez defekata). Greška (ValueError) je: red koji nema 5 cijelih brojeva,
    klasa izvan 1-6, box izvan slike ili s x1 >= x2 / y1 >= y2.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"anotacija ne postoji: {p}")
    try:
        rows = read_ann(p)
    except ValueError as e:
        raise ValueError(f"neispravna anotacija u {p}: {e}") from e
    if any(len(r) != 5 for r in rows):
        raise ValueError(f"neispravna anotacija u {p}: svaki red mora imati 5 vrijednosti (x1 y1 x2 y2 tip)")
    boxes = np.array([r[:4] for r in rows], dtype=float).reshape(-1, 4)
    cls = np.array([r[4] for r in rows], dtype=int)
    if not np.isin(cls, DEFECT_CLASSES).all():
        raise ValueError(f"neispravna anotacija u {p}: klasa mora biti 1-6, dobijeno {sorted(set(cls.tolist()))}")
    bad = ((boxes < 0) | (boxes > IMG_SIZE)).any(axis=1) | (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
    if bad.any():
        raise ValueError(f"neispravna anotacija u {p}: {int(bad.sum())} boxova je izvan slike ili ima x1 >= x2 / y1 >= y2")
    return boxes, cls


def load_pair(item: Item) -> PcbPair:
    """Učitava test i template sliku i anotacije jednog zapisa (iz get_items)."""
    boxes, cls = load_annotations(item["ann"])
    return PcbPair(image_id=image_id(item), group=item["group"], test=load_gray(item["test"]),
                   temp=load_gray(item["temp"]), boxes=boxes, cls=cls)


def iter_pairs(items: Iterable[Item]) -> Iterator[PcbPair]:
    """Generator parova: slike se učitavaju jedna po jedna (cijeli train skup ima oko 600 MB kao uint8)."""
    for item in items:
        yield load_pair(item)


# ----------------------------------------------------------------------------- skupovi i provjere

def get_items(split: str, cfg: ExperimentConfig | None = None) -> list[Item]:
    """Zapisi za 'train', 'val' ili 'test': ista podjela kao u src/evaluation i src/data/to_yolo."""
    cfg = cfg or ExperimentConfig()
    if split not in SPLIT_NAMES:
        raise ValueError(f"nepoznat skup {split!r}; dozvoljeni: {SPLIT_NAMES}")
    root = Path(cfg.data_root)
    split_file = root / "PCBData" / ("test.txt" if split == "test" else "trainval.txt")
    if not split_file.is_file():
        raise FileNotFoundError(f"dataset nije pronađen ({split_file} ne postoji). Pokreni `python scripts/download_data.py` "
                                "ili postavi ExperimentConfig.data_root na folder s DeepPCB-om.")
    items = get_split_items(split, root, cfg.split.val_fraction, cfg.split.seed)
    if not items:
        raise ValueError(f"skup {split!r} je prazan (val_fraction={cfg.split.val_fraction}, seed={cfg.split.seed})")
    return items


def check_items(items: Iterable[Item]) -> list[str]:
    """Lista problema: fajlovi (test, temp, ann) koji ne postoje. Ne otvara slike. Prazna lista = u redu."""
    problems = []
    for item in items:
        for key in ("test", "temp", "ann"):
            if not Path(item[key]).is_file():
                problems.append(f"{image_id(item)}: nedostaje {key} ({item[key]})")
    return problems


def ensure_group_disjoint(first: Iterable[Item], second: Iterable[Item], names: tuple[str, str] = ("train", "val")) -> None:
    """Sprečava curenje podataka: iste ploče (grupe) ne smiju biti u dva skupa. ValueError ako se grupe preklapaju.

    Koristiti za train/val. Za test se NE koristi: 7 od 11 test grupa je po dizajnu i u trainval (docs/PLAN.md).
    """
    overlap = sorted({i["group"] for i in first} & {i["group"] for i in second})
    if overlap:
        raise ValueError(f"grupe {overlap} su i u skupu {names[0]!r} i u skupu {names[1]!r} (curenje podataka)")


def summarize_items(items: Iterable[Item]) -> dict[str, Any]:
    """Sažetak skupa: broj parova, parova po grupi i defekata po klasi (čita i validira anotacije)."""
    groups: Counter[str] = Counter()
    per_class: Counter[int] = Counter()
    n_pairs = 0
    for item in items:
        n_pairs += 1
        groups[item["group"]] += 1
        per_class.update(load_annotations(item["ann"])[1].tolist())
    return {"n_pairs": n_pairs, "groups": dict(sorted(groups.items())),
            "n_defects": sum(per_class.values()),
            "per_class": {NAMES[k]: per_class.get(k, 0) for k in DEFECT_CLASSES}}


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="provjera DeepPCB skupa za Eksperiment 1")
    ap.add_argument("--split", default="val", choices=SPLIT_NAMES)
    ap.add_argument("--root", default=ExperimentConfig().data_root)
    ap.add_argument("--check-images", action="store_true", help="otvori sve slike i provjeri veličinu (sporije)")
    a = ap.parse_args(argv)

    cfg = ExperimentConfig(data_root=a.root)
    items = get_items(a.split, cfg)
    problems = check_items(items)
    if not problems:
        s = summarize_items(items)
        print(f"{a.split}: {s['n_pairs']} parova, {s['n_defects']} defekata, grupe: {s['groups']}")
        print("defekata po klasi:", s["per_class"])
        if a.check_images:
            for item in items:
                load_gray(item["test"]), load_gray(item["temp"])
            print("sve slike su ispravne")
    else:
        print(f"PROBLEMI ({len(problems)}):", *problems[:10], sep="\n  ")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
