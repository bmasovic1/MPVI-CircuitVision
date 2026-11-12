"""Zajednička evaluacija: IoU, uparivanje predikcija s ground truthom, P/R/F1 i mAP.

Format predikcija je opisan u docs/FORMAT.md (JSON, pikselne koordinate 640x640, klase 1-6;
cls = 0 znači detekciju bez klase). Ovaj modul ne zna ništa o modelima, samo čita taj format.

Pravila (ista za sve modele):
- IoU se računa na pikselnim boxovima (širina = x2 - x1, visina = y2 - y1).
- Predikcije se u svakoj slici obrađuju po padajućoj pouzdanosti; svaka se pokušava upariti s
  neuparenim ground truth boxom iste klase koji ima najveći IoU. Uparena je ako je IoU >= prag.
  Jedan GT box se uparuje s najviše jednom predikcijom, svaka ostala je lažna uzbuna (FP).
- AP je površina ispod precision-recall krive (sve tačke, VOC stil), pa se može malo razlikovati
  od vrijednosti koje javlja Ultralytics (koji koristi 101 tačku).
- mAP je prosjek AP-a po klasama koje imaju bar jedan GT box.
"""
import json
from pathlib import Path

import numpy as np

from src.data.deeppcb import DEFAULT_ROOT, NAMES, read_ann, read_split, split_by_group

COCO_IOUS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 ... 0.95
PAPER_IOU = 0.33  # prag iz izvornog rada DeepPCB


# ----------------------------------------------------------------------------- učitavanje

def image_id(item):
    """ID slike: ime bez nastavka i bez _test (npr. '20085000')."""
    return item["ann"].stem


def get_split_items(split="val", root=DEFAULT_ROOT, val_fraction=0.2, seed=42):
    """Zapisi (putanje slika i anotacija) za 'train', 'val' ili 'test'; isti split kao u src/data/to_yolo.py."""
    if split == "test":
        return read_split("test", root)
    train, val = split_by_group(read_split("trainval", root), val_fraction, seed)
    return val if split == "val" else train


def load_ground_truth(split="val", root=DEFAULT_ROOT, val_fraction=0.2, seed=42):
    """Vraća {image_id: (boxes Nx4 float, cls N int)} za 'train', 'val' ili 'test'."""
    gts = {}
    for it in get_split_items(split, root, val_fraction, seed):
        rows = read_ann(it["ann"])
        boxes = np.array([r[:4] for r in rows], dtype=float).reshape(-1, 4)
        cls = np.array([r[4] for r in rows], dtype=int)
        gts[image_id(it)] = (boxes, cls)
    return gts


def load_predictions(path):
    """Čita predictions_<skup>.json i vraća {image_id: (boxes Mx4, scores M, cls M)}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    preds = {}
    for rec in data:
        img = str(rec["image"])
        if img in preds:
            raise ValueError(f"slika {img} se pojavljuje više puta u {path}")
        b = rec["boxes"]
        boxes = np.array([[x["x1"], x["y1"], x["x2"], x["y2"]] for x in b], dtype=float).reshape(-1, 4)
        scores = np.array([x["score"] for x in b], dtype=float)
        cls = np.array([x["cls"] for x in b], dtype=int)
        preds[img] = (boxes, scores, cls)
    return preds


def save_predictions(preds, path):
    """Upis predikcija u formatu iz docs/FORMAT.md (za adaptere modela)."""
    out = []
    for img in sorted(preds):
        boxes, scores, cls = preds[img]
        out.append({
            "image": img,
            "boxes": [
                {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                 "score": float(s), "cls": int(c)}
                for b, s, c in zip(boxes, scores, cls)
            ],
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out), encoding="utf-8")


def prediction_errors(preds):
    """Neispravnosti zbog kojih se rezultati ne smiju računati (prazna lista = u redu)."""
    errs = []
    if not preds:
        return errs
    all_cls = np.concatenate([p[2] for p in preds.values()])
    all_scores = np.concatenate([p[1] for p in preds.values()])
    if len(all_cls) and ((all_cls < 0) | (all_cls > 6)).any():
        errs.append("cls izvan opsega 0-6 (YOLO/RT-DETR klase 0-5 treba pomaknuti za +1).")
    if len(all_cls) and (all_cls == 0).any() and (all_cls != 0).any():
        errs.append("pomiješane klase 0 (bez klase) i 1-6; sve predikcije moraju biti ili cls = 0 ili cls 1-6.")
    if len(all_scores) and (~np.isfinite(all_scores) | (all_scores < 0) | (all_scores > 1)).any():
        errs.append("score nije broj u opsegu 0-1.")
    bad_imgs = []
    for img, (b, _, _) in preds.items():
        if len(b) and (~np.isfinite(b).all() or (b[:, 2] <= b[:, 0]).any() or (b[:, 3] <= b[:, 1]).any()):
            bad_imgs.append(img)
    if bad_imgs:
        errs.append(f"{len(bad_imgs)} slika ima box s x1 >= x2, y1 >= y2 ili vrijednošću koja nije broj (npr. {bad_imgs[:3]}).")
    return errs


def validate_predictions(gts, preds):
    """Prekida s ValueError ako su predikcije neispravne; vraća listu upozorenja (npr. slike bez zapisa)."""
    errs = prediction_errors(preds)
    if errs:
        raise ValueError("Neispravne predikcije:\n  - " + "\n  - ".join(errs))
    warns = []
    missing = sorted(set(gts) - set(preds))
    extra = sorted(set(preds) - set(gts))
    if missing:
        warns.append(f"{len(missing)} slika iz skupa nema zapis u predikcijama (npr. {missing[:3]}); "
                     "računaju se kao bez detekcija. FORMAT.md traži zapis i za slike bez detekcija.")
    if extra:
        warns.append(f"{len(extra)} slika u predikcijama nije u evaluiranom skupu (npr. {extra[:3]}); ignorišu se.")
    return warns


# ----------------------------------------------------------------------------- IoU i uparivanje

def iou_matrix(a, b):
    """IoU svih parova: a je Nx4, b je Mx4 (x1, y1, x2, y2); rezultat je NxM."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def collect_matches(gts, preds, iou_thr, cls=None):
    """Uparuje predikcije s GT-om za jednu klasu (cls=None: sve klase zajedno, bez klase).

    Vraća (scores, is_tp, n_gt): za svaku predikciju njen score i je li pogodak, te broj GT boxova.
    """
    scores, tps, n_gt = [], [], 0
    for img, (gb, gc) in gts.items():
        g = gb if cls is None else gb[gc == cls]
        n_gt += len(g)
        if img not in preds:
            continue
        pb, ps, pc = preds[img]
        if cls is not None:
            m = pc == cls
            pb, ps = pb[m], ps[m]
        if len(ps) == 0:
            continue
        order = np.argsort(-ps, kind="stable")
        pb, ps = pb[order], ps[order]
        ious = iou_matrix(pb, g)
        used = np.zeros(len(g), dtype=bool)
        for i in range(len(pb)):
            hit = False
            if len(g):
                cand = np.where(used, -1.0, ious[i])
                j = int(np.argmax(cand))
                if cand[j] >= iou_thr:
                    used[j] = True
                    hit = True
            scores.append(ps[i])
            tps.append(hit)
    return np.array(scores, float), np.array(tps, bool), n_gt


# ----------------------------------------------------------------------------- AP, P/R/F1

def average_precision(scores, tps, n_gt):
    """AP (površina ispod PR krive, sve tačke). NaN ako klasa nema GT boxova."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(tps[order])
    fp = np.cumsum(~tps[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def _counts(stats, conf):
    """Brojevi TP/FP/FN pri pragu pouzdanosti. stats: {klasa: (scores, tps, n_gt)}.
    Uklanjanje predikcija s niskim scoreom ne mijenja uparivanje onih s visokim, pa je dovoljno filtrirati."""
    out, tot = {}, [0, 0, 0]
    for k, (s, t, n) in stats.items():
        keep = s >= conf
        tp = int(t[keep].sum())
        fp = int(keep.sum()) - tp
        fn = n - tp
        out[k] = (tp, fp, fn, n)
        tot[0] += tp
        tot[1] += fp
        tot[2] += fn
    return out, tuple(tot)


def evaluate(gts, preds, conf=0.25, class_agnostic=None):
    """Računa sve metrike. class_agnostic=None: automatski (True ako su sve predikcije cls=0)."""
    errs = prediction_errors(preds)
    if errs:
        raise ValueError("Neispravne predikcije:\n  - " + "\n  - ".join(errs))
    if class_agnostic is None:
        all_cls = np.concatenate([p[2] for p in preds.values()]) if preds else np.array([], int)
        class_agnostic = len(all_cls) > 0 and bool((all_cls == 0).all())
    keys = [None] if class_agnostic else list(NAMES)

    thrs = sorted(set(COCO_IOUS + [PAPER_IOU]))
    stats = {t: {k: collect_matches(gts, preds, t, k) for k in keys} for t in thrs}

    def mean_ap(thr):
        aps = [average_precision(*stats[thr][k]) for k in keys]
        aps = [a for a in aps if not np.isnan(a)]
        return float(np.mean(aps)) if aps else 0.0

    res = {
        "mode": "bez klase (lokalizacija)" if class_agnostic else "po klasama (6 klasa)",
        "mAP50": mean_ap(0.5),
        "mAP50_95": float(np.mean([mean_ap(t) for t in COCO_IOUS])),
        "mAP33": mean_ap(PAPER_IOU),
        "conf": conf,
    }
    for thr, name in [(0.5, "iou50"), (PAPER_IOU, "iou33")]:
        _, (tp, fp, fn) = _counts(stats[thr], conf)
        p, r, f1 = prf(tp, fp, fn)
        res[name] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}

    best = None  # prag pouzdanosti s najboljim F1 (IoU 0.5), kao što javlja Ultralytics
    for c in np.round(np.arange(0.01, 1.0, 0.01), 2):
        _, (tp, fp, fn) = _counts(stats[0.5], c)
        p, r, f1 = prf(tp, fp, fn)
        if best is None or f1 > best["f1"]:
            best = {"conf": float(c), "precision": p, "recall": r, "f1": f1}
    res["best_f1_iou50"] = best

    per_class_counts, _ = _counts(stats[0.5], conf)
    res["per_class"] = {}
    for k in keys:
        tp, fp, fn, n = per_class_counts[k]
        p, r, f1 = prf(tp, fp, fn)
        res["per_class"][k if k is not None else 0] = {
            "name": "sve" if k is None else NAMES[k], "n_gt": n,
            "ap50": average_precision(*stats[0.5][k]), "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
        }
    return res


def format_report(res):
    """Čitljiv ispis rezultata."""
    lines = [
        f"Način: {res['mode']}",
        f"mAP@0.5 = {res['mAP50']:.4f}   mAP@0.5:0.95 = {res['mAP50_95']:.4f}   mAP@0.33 = {res['mAP33']:.4f}",
        f"Pri conf >= {res['conf']}:",
    ]
    for name, label in [("iou50", "IoU 0.5 "), ("iou33", "IoU 0.33")]:
        r = res[name]
        lines.append(f"  {label}: P = {r['precision']:.4f}  R = {r['recall']:.4f}  F1 = {r['f1']:.4f}"
                     f"   (TP {r['tp']}, FP {r['fp']}, FN {r['fn']})")
    b = res["best_f1_iou50"]
    lines.append(f"Najbolji F1 (IoU 0.5): conf = {b['conf']:.2f}  P = {b['precision']:.4f}  "
                 f"R = {b['recall']:.4f}  F1 = {b['f1']:.4f}")
    lines.append("Po klasi (IoU 0.5):")
    lines.append(f"  {'klasa':10s} {'n_gt':>5s} {'AP50':>7s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    for k, c in res["per_class"].items():
        lines.append(f"  {c['name']:10s} {c['n_gt']:5d} {c['ap50']:7.4f} {c['precision']:7.4f} "
                     f"{c['recall']:7.4f} {c['f1']:7.4f}")
    return "\n".join(lines)
