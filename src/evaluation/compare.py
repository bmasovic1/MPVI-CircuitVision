"""Mjerenje vremena inferencije, upis rezultata u results/results.csv i tabela poređenja eksperimenata.

Upis rezultata jednog eksperimenta iz fajla s predikcijama:
  python -m src.evaluation.compare add --experiment exp3 --variant yolo_template --seed 0 \
      --pred results/exp3/predictions_val.json --split val --ms 12.4 --hardware "Colab T4"
Tabela poređenja i grafovi:
  python -m src.evaluation.compare table --split val --out results

Kolone results.csv prate docs/FORMAT.md. Precision, recall i F1 u njemu su pri pragu pouzdanosti s najboljim F1
(IoU 0.5), da ne zavise od proizvoljnog praga.

Vrijeme inferencije (iz koda adaptera):
  imgs = [ucitana_slika_1, ucitana_slika_2, ...]        # slike unaprijed učitane, učitavanje se ne mjeri
  t = measure_inference_time(lambda x: model(x), imgs)  # batch 1; na GPU-u proslijediti sync=torch.cuda.synchronize
  t["ms_mean"]
"""
import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .metrics import evaluate, load_ground_truth, load_predictions, validate_predictions

COLUMNS = ["experiment", "variant", "seed", "split", "mAP50", "mAP50_95", "mAP33",
           "precision", "recall", "f1", "ms_per_image", "hardware", "commit"]
DEFAULT_CSV = "results/results.csv"


# ----------------------------------------------------------------------------- vrijeme inferencije

def describe_hardware():
    parts = []
    try:
        import torch  # type: ignore 
        if torch.cuda.is_available():
            parts.append(torch.cuda.get_device_name(0))
    except Exception:
        pass
    parts.append(platform.processor() or platform.machine())
    return ", ".join(p for p in parts if p)


def measure_inference_time(predict, images, warmup=10, runs=100, sync=None):
    """Prosječno vrijeme jedne inferencije u ms. Protokol za sve modele: batch 1, 10 zagrijavanja, prosjek 100 slika.

    predict: funkcija koja prima jedan element iz images; images: unaprijed učitani ulazi (ako ih je manje
    od 'runs', ciklično se ponavljaju); sync: funkcija za sinhronizaciju GPU-a (npr. torch.cuda.synchronize).
    """
    n = len(images)
    for i in range(warmup):
        predict(images[i % n])
    if sync:
        sync()
    times = []
    for i in range(runs):
        x = images[i % n]
        t0 = time.perf_counter()
        predict(x)
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000)
    return {"ms_mean": float(np.mean(times)), "ms_std": float(np.std(times)), "ms_median": float(np.median(times)),
            "runs": runs, "warmup": warmup, "hardware": describe_hardware()}


# ----------------------------------------------------------------------------- results.csv

def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def result_row(experiment, variant, seed, split, res, ms_per_image=None, hardware=None):
    """Red za results.csv iz rezultata funkcije evaluate()."""
    best = res["best_f1_iou50"]
    return {
        "experiment": experiment, "variant": variant, "seed": seed, "split": split,
        "mAP50": round(res["mAP50"], 4), "mAP50_95": round(res["mAP50_95"], 4), "mAP33": round(res["mAP33"], 4),
        "precision": round(best["precision"], 4), "recall": round(best["recall"], 4), "f1": round(best["f1"], 4),
        "ms_per_image": "" if ms_per_image is None else round(float(ms_per_image), 2),
        "hardware": hardware or "", "commit": _git_commit(),
    }


def append_result(row, path=DEFAULT_CSV):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLUMNS})


def read_results(path=DEFAULT_CSV):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ["mAP50", "mAP50_95", "mAP33", "precision", "recall", "f1", "ms_per_image"]:
                r[k] = float(r[k]) if r.get(k, "") != "" else None
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------- tabela poređenja

METRICS = [("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95"), ("mAP33", "mAP@0.33"),
           ("precision", "P"), ("recall", "R"), ("f1", "F1"), ("ms_per_image", "ms/slika")]


def aggregate(rows, split):
    """Grupiše po (eksperiment, varijanta) i računa prosjek i odstupanje preko seedova."""
    groups = {}
    for r in rows:
        if r["split"] == split:
            groups.setdefault((r["experiment"], r["variant"]), []).append(r)
    out = []
    for (exp, var), rs in groups.items():
        entry = {"experiment": exp, "variant": var, "n": len(rs),
                 "hardware": sorted({r["hardware"] for r in rs if r["hardware"]})}
        for key, _ in METRICS:
            vals = [r[key] for r in rs if r[key] is not None]
            entry[key] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else None) if vals else None
        out.append(entry)
    return sorted(out, key=lambda e: -(e["mAP50"][0] if e["mAP50"] else -1))


def _fmt(v, digits=3):
    if v is None:
        return "-"
    m, s = v
    return f"{m:.{digits}f}" if s is None else f"{m:.{digits}f} ± {s:.{digits}f}"


def comparison_table(rows, split):
    """Markdown tabela poređenja (prosjek ± odstupanje preko seedova)."""
    agg = aggregate(rows, split)
    head = ["Eksperiment", "Varijanta", "seedova"] + [n for _, n in METRICS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for e in agg:
        cells = [e["experiment"], e["variant"], str(e["n"])]
        cells += [_fmt(e[k], 1 if k == "ms_per_image" else 3) for k, _ in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    hw = sorted({h for e in agg for h in e["hardware"]})
    notes = [f"\nSkup: `{split}`. Vrijednosti su prosjek ± odstupanje preko seedova; P, R i F1 su pri pragu pouzdanosti "
             "s najboljim F1 (IoU 0.5). Razlike manje od odstupanja između seedova ne treba tumačiti kao poboljšanje."]
    if hw:
        notes.append("Hardver: " + "; ".join(hw) + ".")
    if len(hw) > 1:
        notes.append("**Upozorenje:** vrijeme je mjereno na različitom hardveru, pa ms/slika nije uporediv.")
    return "\n".join(lines) + "\n" + "\n".join(notes) + "\n"


def plot_comparison(rows, split, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = [e for e in aggregate(rows, split) if e["mAP50"]]
    if not agg:
        return
    names = [f"{e['experiment']}\n{e['variant']}" for e in agg]
    means = [e["mAP50"][0] for e in agg]
    errs = [e["mAP50"][1] or 0 for e in agg]
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(agg)), 4.2))
    ax.bar(range(len(agg)), means, yerr=errs, capsize=4, color="#4a78b5")
    ax.set_xticks(range(len(agg)), names, fontsize=8)
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(max(0, min(means) - 0.1), 1.0)
    ax.set_title(f"Poređenje eksperimenata ({split})")
    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out_dir) / f"comparison_map_{split}.png", dpi=130)
    plt.close(fig)

    timed = [e for e in agg if e["ms_per_image"]]
    if timed:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for e in timed:
            ax.scatter(e["ms_per_image"][0], e["mAP50"][0])
            ax.annotate(f"{e['experiment']}/{e['variant']}", (e["ms_per_image"][0], e["mAP50"][0]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("ms po slici")
        ax.set_ylabel("mAP@0.5")
        ax.set_title(f"Tačnost naspram brzine ({split})")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"comparison_speed_{split}.png", dpi=130)
        plt.close(fig)


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("add", help="izračunaj metrike iz predikcija i upiši red u results.csv")
    a_.add_argument("--experiment", required=True)
    a_.add_argument("--variant", required=True)
    a_.add_argument("--seed", type=int, default=0)
    a_.add_argument("--pred", required=True)
    a_.add_argument("--split", default="val", choices=["val", "test"])
    a_.add_argument("--root", default="data/DeepPCB")
    a_.add_argument("--conf", type=float, default=0.25)
    a_.add_argument("--ms", type=float, help="vrijeme inferencije po slici u ms")
    a_.add_argument("--hardware", help="npr. 'Colab T4'")
    a_.add_argument("--csv", default=DEFAULT_CSV)
    t_ = sub.add_parser("table", help="tabela poređenja i grafovi iz results.csv")
    t_.add_argument("--csv", default=DEFAULT_CSV)
    t_.add_argument("--split", default="val", choices=["val", "test"])
    t_.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.cmd == "add":
        gts = load_ground_truth(a.split, a.root)
        preds = load_predictions(a.pred)
        try:
            for w in validate_predictions(gts, preds):
                print("UPOZORENJE:", w)
            res = evaluate(gts, preds, conf=a.conf)
        except ValueError as e:
            raise SystemExit(f"GREŠKA: {e}")
        row = result_row(a.experiment, a.variant, a.seed, a.split, res, a.ms, a.hardware)
        append_result(row, a.csv)
        print(f"Upisano u {a.csv}: mAP@0.5 = {row['mAP50']}, mAP@0.5:0.95 = {row['mAP50_95']}, F1 = {row['f1']}")
    else:
        rows = read_results(a.csv)
        md = comparison_table(rows, a.split)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"comparison_{a.split}.md").write_text(md, encoding="utf-8")
        plot_comparison(rows, a.split, out)
        print(md)


if __name__ == "__main__":
    main()
