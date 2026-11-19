"""Označavanje kandidata (klasa 1-6 ili pozadina) i dijagnostika dosega lokalizacije.

Oznaka kandidata = klasa GT boxa s najvećim IoU, ako je IoU >= `label_iou` (početno 0.5), inače BACKGROUND (0).
Svi kandidati koji pređu prag postaju pozitivni primjeri te klase (duplikate iste regije na izlazu uklanja NMS u koraku 8).
Oznake se koriste samo za TRENIRANJE i dijagnostiku; pri predikciji ground truth se ne koristi.

Recall lokalizacije = udio GT defekata za koje postoji bar jedan kandidat s IoU >= prag (bez obzira na klasu).
To je gornja granica onoga što klasifikator može pronaći: defekt bez kandidata se ne može detektovati.

Dijagnostika (train ili val; test se namjerno ne nudi, da se parametri ne podešavaju na njemu):
    python -m src.experiment_1_classical.labeling --split val
    python -m src.experiment_1_classical.labeling --split train --step 5     # svaka 5. slika (brže)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Iterable

import numpy as np

from src.data.deeppcb import NAMES
from src.evaluation.metrics import PAPER_IOU, iou_matrix

from .candidates import Candidates, extract_candidates
from .config import BACKGROUND, DEFECT_CLASSES, ExperimentConfig
from .dataset import PcbPair, get_items, iter_pairs

RECALL_IOUS = (0.5, PAPER_IOU)  # IoU 0.5 kao glavna metrika i 0.33 kao u izvornom DeepPCB radu


def label_candidates(cand_boxes: np.ndarray, gt_boxes: np.ndarray, gt_cls: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Oznake kandidata (N,): klasa 1-6 najbližeg GT boxa ako je IoU >= iou_thr, inače BACKGROUND."""
    cand_boxes = np.asarray(cand_boxes, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    gt_cls = np.asarray(gt_cls, dtype=int).reshape(-1)
    if len(gt_boxes) != len(gt_cls):
        raise ValueError(f"broj GT boxova ({len(gt_boxes)}) i klasa ({len(gt_cls)}) se ne poklapa")
    if not 0.0 < iou_thr <= 1.0:
        raise ValueError(f"iou_thr mora biti u (0, 1], dobijeno {iou_thr!r}")
    labels = np.full(len(cand_boxes), BACKGROUND, dtype=int)
    if len(cand_boxes) == 0 or len(gt_boxes) == 0:
        return labels
    iou = iou_matrix(cand_boxes, gt_boxes)  # (N, M)
    best = iou.argmax(axis=1)
    hit = iou[np.arange(len(best)), best] >= iou_thr
    labels[hit] = gt_cls[best[hit]]
    return labels


def label_pair(pair: PcbPair, cfg: ExperimentConfig | None = None) -> tuple[Candidates, np.ndarray]:
    """Kandidati para slika i njihove oznake (za skupove za treniranje)."""
    cfg = cfg or ExperimentConfig()
    cand = extract_candidates(pair.test, pair.temp, cfg.candidates)
    return cand, label_candidates(cand.boxes, pair.boxes, pair.cls, cfg.candidates.label_iou)


def candidate_stats(pairs: Iterable[PcbPair], cfg: ExperimentConfig | None = None,
                    iou_thresholds: tuple[float, ...] = RECALL_IOUS) -> dict:
    """Statistika kandidata: broj, udio pozitivnih, raspodjela oznaka i recall lokalizacije po klasi i pragu IoU."""
    cfg = cfg or ExperimentConfig()
    n_images = n_cand = 0
    n_gt: Counter[int] = Counter()
    covered = {t: Counter() for t in iou_thresholds}
    label_counts: Counter[int] = Counter()
    for pair in pairs:
        cand, labels = label_pair(pair, cfg)
        n_images += 1
        n_cand += len(cand)
        label_counts.update(labels.tolist())
        n_gt.update(pair.cls.tolist())
        iou = iou_matrix(pair.boxes, cand.boxes)  # (M, N)
        best = iou.max(axis=1) if iou.shape[1] else np.zeros(len(pair.boxes))
        for t in iou_thresholds:
            covered[t].update(pair.cls[best >= t].tolist())
    total = sum(n_gt.values())
    recall = {}
    for t in iou_thresholds:
        per_class = {NAMES[k]: (covered[t][k] / n_gt[k] if n_gt[k] else float("nan")) for k in DEFECT_CLASSES}
        recall[t] = {"all": sum(covered[t].values()) / total if total else float("nan"), **per_class}
    return {
        "n_images": n_images, "n_gt": total, "n_candidates": n_cand,
        "candidates_per_image": n_cand / n_images if n_images else float("nan"),
        "positive_fraction": 1 - label_counts[BACKGROUND] / n_cand if n_cand else float("nan"),
        "label_counts": {"pozadina": label_counts[BACKGROUND], **{NAMES[k]: label_counts[k] for k in DEFECT_CLASSES}},
        "recall": recall,
    }


def format_stats(stats: dict, title: str = "") -> str:
    """Čitljiv ispis rezultata candidate_stats."""
    lines = [f"{title}{stats['n_images']} slika, {stats['n_gt']} defekata, {stats['n_candidates']} kandidata "
             f"({stats['candidates_per_image']:.1f} po slici, pozitivnih {stats['positive_fraction']:.1%})",
             "Oznake kandidata: " + ", ".join(f"{k} {v}" for k, v in stats["label_counts"].items()),
             "Recall lokalizacije (udio defekata s bar jednim kandidatom IoU >= prag):",
             f"  {'IoU':>5s} {'ukupno':>8s} " + " ".join(f"{NAMES[k]:>9s}" for k in DEFECT_CLASSES)]
    for t, r in stats["recall"].items():
        lines.append(f"  {t:5.2f} {r['all']:8.1%} " + " ".join(f"{r[NAMES[k]]:9.1%}" for k in DEFECT_CLASSES))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="dijagnostika kandidata: recall lokalizacije i raspodjela oznaka")
    ap.add_argument("--split", default="val", choices=["train", "val"])  # test se ne koristi za podešavanje parametara
    ap.add_argument("--root", default=ExperimentConfig().data_root)
    ap.add_argument("--step", type=int, default=1, help="uzmi svaku N-tu sliku (brža provjera)")
    a = ap.parse_args(argv)
    if a.step < 1:
        ap.error("--step mora biti >= 1")
    cfg = ExperimentConfig(data_root=a.root)
    items = get_items(a.split, cfg)[::a.step]
    print(format_stats(candidate_stats(iter_pairs(items), cfg), title=f"{a.split}: "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
