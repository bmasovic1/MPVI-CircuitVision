"""Analiza grešaka: matrica konfuzije, najčešće zamjene klasa, recall po veličini defekta, vizualizacija.

Pokretanje:
  python -m src.evaluation.analysis --pred results/exp3_yolo/predictions_val.json --split val --out results/exp3_yolo/analysis
Rezultat u --out: analysis.txt, confusion_matrix.png i errors/<slika>.png (slike s najviše grešaka).
"""
import argparse
from pathlib import Path

import numpy as np

from .metrics import NAMES, get_split_items, image_id, iou_matrix, load_ground_truth, load_predictions

BG = 6  # indeks "pozadine" u matrici konfuzije; klase 1-6 su indeksi 0-5
LABELS = [NAMES[i] for i in range(1, 7)] + ["pozadina"]
SIZE_BINS = [0, 30, 40, 50, float("inf")]  # veličina = sqrt(širina * visina) u pikselima


def match_detail(gts, preds, iou_thr=0.5, conf=0.25, class_aware=False):
    """Uparuje predikcije (score >= conf) s ground truthom, slika po slika, po padajućem scoreu.

    Vraća (gt_rows, pred_rows):
      gt_rows:   dict(img, cls, size, pred_cls)  pred_cls = klasa uparene predikcije ili -1 (promašen defekt, FN)
      pred_rows: dict(img, idx, cls, score, gt_cls)  gt_cls = klasa uparenog GT-a ili -1 (lažna uzbuna, FP)
    class_aware=False: predikcija se upari s najboljim GT-om bilo koje klase, pa se pogrešna klasa vidi kao
    zamjena (a ne kao FP + FN). To je potrebno za matricu konfuzije.
    """
    gt_rows, pred_rows = [], []
    for img, (gb, gc) in gts.items():
        w = np.clip(gb[:, 2] - gb[:, 0], 0, None) if len(gc) else np.zeros(0)
        h = np.clip(gb[:, 3] - gb[:, 1], 0, None) if len(gc) else np.zeros(0)
        size = np.sqrt(w * h)
        rows = [dict(img=img, cls=int(gc[i]), size=float(size[i]), pred_cls=-1) for i in range(len(gc))]
        if img in preds:
            pb, ps, pc = preds[img]
            keep = ps >= conf
            idx = np.flatnonzero(keep)  # indeksi u ORIGINALNOM nizu predikcija (za crtanje)
            pb, ps, pc = pb[keep], ps[keep], pc[keep]
            order = np.argsort(-ps, kind="stable")
            ious = iou_matrix(pb[order], gb)
            used = np.zeros(len(gc), dtype=bool)
            for o, i in enumerate(order):
                gt_i = -1
                if len(gc):
                    cand = np.where(used, -1.0, ious[o])
                    if class_aware:
                        cand = np.where(gc == pc[i], cand, -1.0)
                    j = int(np.argmax(cand))
                    if cand[j] >= iou_thr:
                        gt_i = j
                        used[j] = True
                if gt_i >= 0:
                    rows[gt_i]["pred_cls"] = int(pc[i])
                pred_rows.append(dict(img=img, idx=int(idx[i]), cls=int(pc[i]), score=float(ps[i]),
                                      gt_cls=rows[gt_i]["cls"] if gt_i >= 0 else -1, gt_idx=gt_i))
        gt_rows.extend(rows)
    return gt_rows, pred_rows


def confusion_matrix(gt_rows, pred_rows):
    """7x7: redovi = stvarna klasa (+ pozadina), kolone = predviđena klasa (+ pozadina).
    Pozadina u koloni = promašen defekt (FN); pozadina u redu = lažna uzbuna (FP)."""
    cm = np.zeros((7, 7), dtype=int)
    for g in gt_rows:
        cm[g["cls"] - 1, g["pred_cls"] - 1 if g["pred_cls"] > 0 else BG] += 1
    for p in pred_rows:
        if p["gt_cls"] < 0:
            cm[BG, p["cls"] - 1] += 1
    return cm


def top_confusions(cm, n=5):
    """Najčešće zamjene klasa: lista (stvarna, predviđena, broj), bez pozadine."""
    pairs = [(NAMES[i + 1], NAMES[j + 1], int(cm[i, j])) for i in range(6) for j in range(6)
             if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda x: -x[2])[:n]


def recall_by_size(gt_rows, bins=SIZE_BINS):
    """Udio tačno detektovanih (uparenih s istom klasom) defekata po veličini."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        rows = [g for g in gt_rows if lo <= g["size"] < hi]
        ok = sum(1 for g in rows if g["pred_cls"] == g["cls"])
        label = f"< {hi:.0f} px" if lo == 0 else (f">= {lo:.0f} px" if hi == float("inf") else f"{lo:.0f}-{hi:.0f} px")
        out.append({"label": label, "n": len(rows), "recall": ok / len(rows) if rows else float("nan")})
    return out


def analyze(gts, preds, iou_thr=0.5, conf=0.25):
    all_cls = np.concatenate([p[2] for p in preds.values()]) if preds else np.array([], int)
    if len(all_cls) and (all_cls == 0).all():
        raise ValueError("analiza grešaka traži predikcije s klasama 1-6 (cls = 0 nema klasu)")
    gt_rows, pred_rows = match_detail(gts, preds, iou_thr, conf, class_aware=False)
    cm = confusion_matrix(gt_rows, pred_rows)
    n_gt = len(gt_rows)
    correct = sum(1 for g in gt_rows if g["pred_cls"] == g["cls"])
    wrong = sum(1 for g in gt_rows if g["pred_cls"] > 0 and g["pred_cls"] != g["cls"])
    return {
        "iou_thr": iou_thr, "conf": conf, "confusion_matrix": cm.tolist(), "n_gt": n_gt,
        "correct": correct, "wrong_class": wrong, "missed": n_gt - correct - wrong,
        "false_positives": sum(1 for p in pred_rows if p["gt_cls"] < 0),
        "top_confusions": top_confusions(cm), "recall_by_size": recall_by_size(gt_rows),
        "gt_rows": gt_rows, "pred_rows": pred_rows,
    }


def format_analysis(an):
    cm = np.array(an["confusion_matrix"])
    lines = [f"Analiza grešaka (IoU {an['iou_thr']}, conf >= {an['conf']})", "",
             "Matrica konfuzije (redovi = stvarna klasa, kolone = predviđena klasa):",
             "  " + " " * 10 + "".join(f"{name[:9]:>10s}" for name in LABELS)]
    for i, name in enumerate(LABELS):
        lines.append(f"  {name:10s}" + "".join(f"{v:10d}" for v in cm[i]))
    lines.append("  (kolona 'pozadina' = promašen defekt, red 'pozadina' = lažna uzbuna)")
    n = an["n_gt"]
    lines += ["", f"Defekata ukupno: {n}",
              f"  tačno detektovano:    {an['correct']} ({an['correct'] / n:.1%})" if n else "",
              f"  pogrešna klasa:       {an['wrong_class']}",
              f"  promašeno (FN):       {an['missed']}",
              f"Lažne uzbune (FP):      {an['false_positives']}", "", "Najčešće zamjene klasa (stvarna -> predviđena):"]
    lines += [f"  {a} -> {b}: {c}" for a, b, c in an["top_confusions"]] or ["  nema"]
    lines += ["", "Tačno detektovani defekti po veličini (sqrt(širina*visina)):"]
    lines += [f"  {r['label']:12s} n = {r['n']:4d}   {r['recall']:.1%}" if r["n"] else f"  {r['label']:12s} n = 0"
              for r in an["recall_by_size"]]
    return "\n".join(lines)


def plot_confusion_matrix(an, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(an["confusion_matrix"])
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(7), LABELS)
    ax.set_xlabel("predviđeno")
    ax.set_ylabel("stvarno")
    ax.set_title("Matrica konfuzije (boja: udio reda, broj: defekata)")
    for i in range(7):
        for j in range(7):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def worst_images(an, n=6):
    """ID-evi slika s najviše grešaka (promašeni + pogrešna klasa + lažne uzbune)."""
    err = {}
    for g in an["gt_rows"]:
        if g["pred_cls"] != g["cls"]:
            err[g["img"]] = err.get(g["img"], 0) + 1
    for p in an["pred_rows"]:
        if p["gt_cls"] < 0:
            err[p["img"]] = err.get(p["img"], 0) + 1
    return [k for k, _ in sorted(err.items(), key=lambda kv: -kv[1])[:n]]


def draw_errors(item, gts, preds, path, iou_thr=0.5, conf=0.25, scale=2):
    """Crta jednu sliku: zeleno = tačno detektovan defekt, crveno = promašen ili pogrešna klasa (stvarni box),
    žuto = lažna uzbuna, ljubičasto = predikcija s pogrešnom klasom."""
    from PIL import Image, ImageDraw
    img_id = image_id(item)
    gb, gc = gts[img_id]
    g1 = {img_id: gts[img_id]}
    p1 = {img_id: preds[img_id]} if img_id in preds else {}
    gt_rows, pred_rows = match_detail(g1, p1, iou_thr, conf, class_aware=False)
    im = Image.open(item["test"]).convert("RGB")
    im = im.resize((im.width * scale, im.height * scale))
    d = ImageDraw.Draw(im)
    for box, row in zip(gb, gt_rows):
        ok = row["pred_cls"] == row["cls"]
        d.rectangle([v * scale for v in box], outline="lime" if ok else "red", width=2)
        d.text((box[0] * scale, max(0, box[1] * scale - 11)), NAMES[row["cls"]], fill="lime" if ok else "red")
    pb = preds[img_id][0] if img_id in preds else np.zeros((0, 4))
    for pr in pred_rows:
        if pr["gt_cls"] < 0 or pr["gt_cls"] != pr["cls"]:
            col = "yellow" if pr["gt_cls"] < 0 else "magenta"
            b = pb[pr["idx"]]
            d.rectangle([v * scale for v in b], outline=col, width=1)
            d.text((b[0] * scale, b[3] * scale + 1), f"{NAMES[pr['cls']]} {pr['score']:.2f}", fill=col)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--root", default="data/DeepPCB")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--n-examples", type=int, default=6)
    a = ap.parse_args()

    gts = load_ground_truth(a.split, a.root)
    preds = load_predictions(a.pred)
    an = analyze(gts, preds, a.iou, a.conf)
    text = format_analysis(an)
    print(text)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.txt").write_text(text, encoding="utf-8")
    plot_confusion_matrix(an, out / "confusion_matrix.png")
    items = {image_id(i): i for i in get_split_items(a.split, a.root)}
    for img_id in worst_images(an, a.n_examples):
        draw_errors(items[img_id], gts, preds, out / "errors" / f"{img_id}.png", a.iou, a.conf)
    print(f"\nZapisano u {out}/")


if __name__ == "__main__":
    main()
