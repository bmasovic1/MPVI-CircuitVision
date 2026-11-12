"""Robusnost modela: kako se metrike mijenjaju kad se test slika pokvari.

Poremećaji se primjenjuju samo na TEST sliku; template slika ostaje netaknuta:
  noise (Gaussov šum, sigma u nivoima sive 0-255), blur (radijus), contrast, brightness (faktor),
  shift (pomak test slike u pikselima po x u odnosu na template). Kod pomaka se i ground truth pomiče
  zajedno sa slikom (defekt je na novom mjestu), pa se mjeri koliko model gubi zbog neporavnanja.

Adapter modela daje funkciju:
  def predict(test_img, temp_img, image_id):
      # test_img, temp_img: numpy uint8 (640, 640), sivi tonovi
      # image_id služi samo za logiranje; model ga ne smije koristiti
      return boxes (N,4: x1,y1,x2,y2 u pikselima), scores (N), cls (N, klase 1-6 ili 0)
Primjer:
  from src.evaluation.robustness import run_robustness
  run_robustness(predict, experiment="exp3", variant="yolo_template", split="val",
                 out_csv="results/robustness_exp3_yolo_template.csv")
Grafovi:  python -m src.evaluation.robustness --csv results/robustness_*.csv --out results/robustness.png
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .metrics import DEFAULT_ROOT, evaluate, get_split_items, image_id, load_ground_truth

IMG_SIZE = 640


# ----------------------------------------------------------------------------- poremećaji

def add_noise(img, sigma, rng):
    out = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(img, radius):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)))


def change_contrast(img, factor):
    return np.array(ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor))


def change_brightness(img, factor):
    return np.array(ImageEnhance.Brightness(Image.fromarray(img)).enhance(factor))


def shift_image(img, dx, dy=0):
    """Pomjera sadržaj za (dx, dy) piksela (dx > 0: udesno, dy > 0: nadole); ivice se popunjavaju ponavljanjem."""
    h, w = img.shape[:2]
    pad = max(abs(dx), abs(dy))
    p = np.pad(img, pad, mode="edge")
    return p[pad - dy: pad - dy + h, pad - dx: pad - dx + w]


def shift_gts(gts, dx, dy=0, size=IMG_SIZE):
    """Pomjera ground truth zajedno sa slikom; boxovi se sijeku na granice slike, a ispali se izbacuju."""
    out = {}
    for img, (b, c) in gts.items():
        nb = b + np.array([dx, dy, dx, dy], dtype=float)
        nb = np.clip(nb, 0, size)
        keep = (nb[:, 2] - nb[:, 0] > 1) & (nb[:, 3] - nb[:, 1] > 1)
        out[img] = (nb[keep], c[keep])
    return out


PERTURBATIONS = {
    "noise": {"levels": [5, 10, 20], "fn": lambda im, lv, rng: add_noise(im, lv, rng)},
    "blur": {"levels": [0.5, 1.0, 2.0], "fn": lambda im, lv, rng: blur(im, lv)},
    "contrast": {"levels": [0.5, 0.75, 1.5], "fn": lambda im, lv, rng: change_contrast(im, lv)},
    "brightness": {"levels": [0.7, 1.3], "fn": lambda im, lv, rng: change_brightness(im, lv)},
    "shift": {"levels": [1, 2, 4, 8], "fn": lambda im, lv, rng: shift_image(im, int(lv), 0),
              "gt": lambda gts, lv: shift_gts(gts, int(lv), 0)},
}
COLUMNS = ["experiment", "variant", "perturbation", "level", "mAP50", "mAP50_95", "mAP33", "f1", "delta_mAP50"]


def _load_gray(path):
    return np.array(Image.open(path).convert("L"))


def run_robustness(predict_fn, experiment, variant, split="val", root=DEFAULT_ROOT, out_csv=None,
                   perturbations=None, limit=None, seed=0, conf=0.25):
    """Vrti model na neizmijenjenim i pokvarenim slikama i vraća listu redova (opcionalno upisuje CSV)."""
    perturbations = perturbations or PERTURBATIONS
    items = get_split_items(split, root)
    if limit:
        items = items[:limit]
    ids = {image_id(i) for i in items}
    gts = {k: v for k, v in load_ground_truth(split, root).items() if k in ids}

    conditions = [("clean", 0)] + [(n, lv) for n, p in perturbations.items() for lv in p["levels"]]
    rows, clean_map = [], None
    for name, level in conditions:
        spec = perturbations.get(name)
        gts_c = spec["gt"](gts, level) if spec and "gt" in spec else gts
        preds = {}
        for k, it in enumerate(items):
            test, temp = _load_gray(it["test"]), _load_gray(it["temp"])
            if spec:
                test = spec["fn"](test, level, np.random.default_rng(seed * 100003 + k))
            b, s, c = predict_fn(test, temp, image_id(it))
            preds[image_id(it)] = (np.asarray(b, float).reshape(-1, 4), np.asarray(s, float).reshape(-1),
                                   np.asarray(c, int).reshape(-1))
        res = evaluate(gts_c, preds, conf=conf)
        if name == "clean":
            clean_map = res["mAP50"]
        rows.append({"experiment": experiment, "variant": variant, "perturbation": name, "level": level,
                     "mAP50": round(res["mAP50"], 4), "mAP50_95": round(res["mAP50_95"], 4),
                     "mAP33": round(res["mAP33"], 4), "f1": round(res["best_f1_iou50"]["f1"], 4),
                     "delta_mAP50": round(res["mAP50"] - clean_map, 4)})
        print(f"  {name:10s} {level:>5}: mAP@0.5 = {res['mAP50']:.4f} ({res['mAP50'] - clean_map:+.4f})")
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
    return rows


def plot_robustness(csv_paths, out_png, metric="mAP50"):
    """Jedan graf po vrsti poremećaja; linija po modelu (eksperiment/varijanta)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {}
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                data.setdefault(f"{r['experiment']}/{r['variant']}", []).append(r)
    kinds = [k for k in PERTURBATIONS]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.6), sharey=True)
    for ax, kind in zip(axes, kinds):
        for model, rs in data.items():
            clean = [r for r in rs if r["perturbation"] == "clean"]
            pts = sorted([r for r in rs if r["perturbation"] == kind], key=lambda r: float(r["level"]))
            if not pts:
                continue
            xs = [0.0] + [float(r["level"]) for r in pts]
            ys = [float(clean[0][metric]) if clean else np.nan] + [float(r[metric]) for r in pts]
            ax.plot(xs, ys, marker="o", label=model)
        ax.set_title(kind + (" (px)" if kind == "shift" else ""))
        ax.set_xlabel("jačina")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(metric)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="graf robusnosti iz CSV fajlova koje je napravio run_robustness")
    ap.add_argument("--csv", nargs="+", required=True, help="fajlovi ili wildcard, npr. results/robustness_*.csv")
    ap.add_argument("--out", default="results/robustness.png")
    ap.add_argument("--metric", default="mAP50", choices=["mAP50", "mAP50_95", "mAP33", "f1"])
    a = ap.parse_args()
    paths = [q for p in a.csv for q in glob.glob(p)]
    plot_robustness(paths, a.out, a.metric)
    print("Zapisano:", a.out)


if __name__ == "__main__":
    main()
