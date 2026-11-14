"""Istraživačka analiza podataka (EDA) nad DeepPCB: statistike i slike za docs/DATASET.md.

Pokretanje (potreban je preuzet dataset, python scripts/download_data.py):
  python -m src.data.eda
Piše docs/figures/*.png i docs/figures/eda_stats.json; isti brojevi se ispisuju u terminalu.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .deeppcb import DEFAULT_ROOT, NAMES, read_ann, read_split, split_by_group

IMG = 640
PX_PER_MM = 48  # prema README-u DeepPCB: oko 48 piksela po milimetru
DIFF_THR = 100  # prag za "piksel se razlikuje" na slikama 0-255
MARGIN = 12  # ivica koja se zanemaruje pri procjeni pomaka (np.roll vraća sadržaj s druge strane)


def load_items(root):
    """Zapisi s anotacijama: id, grupa, niz boxova (n, 5), putanje."""
    out = {}
    for split in ["trainval", "test"]:
        recs = []
        for it in read_split(split, root):
            rows = np.array(read_ann(it["ann"]), dtype=int).reshape(-1, 5)
            recs.append({**it, "id": it["ann"].stem, "rows": rows})
        out[split] = recs
    return out


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def estimate_shift(test, temp):
    """Fazna korelacija: pomak (dx, dy) koji treba primijeniti na template (np.roll) da se poklopi s test slikom."""
    a = test.astype(np.float32) - test.mean()
    b = temp.astype(np.float32) - temp.mean()
    r = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    r /= np.abs(r) + 1e-9
    c = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    dy = iy if iy < IMG // 2 else iy - IMG
    dx = ix if ix < IMG // 2 else ix - IMG
    return int(dx), int(dy)


def diff_in_interior(a, b):
    return float((np.abs(a.astype(np.int16) - b.astype(np.int16))[MARGIN:-MARGIN, MARGIN:-MARGIN] > DIFF_THR).mean())


def pair_stats(rec):
    """Svojstva jednog para: tamni dio slike, razlika izvan boxova, procjena pomaka."""
    t = np.array(Image.open(rec["test"]).convert("L"))
    p = np.array(Image.open(rec["temp"]).convert("L"))
    inside = np.zeros(t.shape, bool)
    for x1, y1, x2, y2, _ in rec["rows"]:
        inside[y1:y2, x1:x2] = True
    d = np.abs(t.astype(np.int16) - p.astype(np.int16)) > DIFF_THR
    dx, dy = estimate_shift(t, p)
    before = diff_in_interior(t, p)
    after = diff_in_interior(t, np.roll(p, (dy, dx), axis=(0, 1)))
    edges = int((np.abs(np.diff(t.astype(np.int16), axis=1)) > DIFF_THR).sum())
    return {
        "dark": float((t < 128).mean()),
        "extreme": float(((t < 10) | (t > 245)).mean()),
        "diff": int(d.sum()), "diff_inside": int((d & inside).sum()), "diff_outside": int((d & ~inside).sum()),
        "shift": (dx, dy), "before": before, "after": after, "edges": edges,
        "mode": Image.open(rec["test"]).mode, "size": t.shape,
    }


def box_table(recs):
    """Niz (N, 8): x1, y1, x2, y2, klasa, w, h, indeks slike."""
    rows = []
    for k, r in enumerate(recs):
        for x1, y1, x2, y2, c in r["rows"]:
            rows.append([x1, y1, x2, y2, c, x2 - x1, y2 - y1, k])
    return np.array(rows, dtype=float).reshape(-1, 8)


def describe(v):
    v = np.asarray(v, float)
    return {"mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2), "min": round(float(v.min()), 2),
            "p5": round(float(np.percentile(v, 5)), 2), "p95": round(float(np.percentile(v, 95)), 2),
            "max": round(float(v.max()), 2)}


def compute(root):
    data = load_items(root)
    tv, te = data["trainval"], data["test"]
    train, val = split_by_group(tv)
    # --- grupe i šabloni
    pcb = Path(root) / "PCBData"
    all_temp = {p.name for p in pcb.glob("*/*/*_temp.jpg")}
    used_temp = {r["temp"].name for r in tv + te}
    groups = collections.defaultdict(lambda: {"trainval": 0, "test": 0, "boxes": collections.Counter()})
    for split, recs in [("trainval", tv), ("test", te)]:
        for r in recs:
            groups[r["group"]][split] += 1
            groups[r["group"]]["boxes"].update(r["rows"][:, 4].tolist())
    temp_hash = {r["temp"].name: md5(r["temp"]) for r in tv + te}
    test_hash = collections.Counter(md5(r["test"]) for r in tv + te)
    per_group_unique = collections.defaultdict(set)
    for r in tv + te:
        per_group_unique[r["group"]].add(temp_hash[r["temp"].name])

    st = {"pairs": len(tv) + len(te), "trainval": len(tv), "test": len(te), "val": len(val), "train": len(train),
          "temp_files_total": len(all_temp), "temp_files_unused": sorted(all_temp - used_temp),
          "unique_templates": len(set(temp_hash.values())), "duplicate_test_images": int(sum(v - 1 for v in test_hash.values())),
          "groups": {g: {"trainval": v["trainval"], "test": v["test"], "unique_templates": len(per_group_unique[g]),
                         "boxes": {NAMES[k]: v["boxes"].get(k, 0) for k in NAMES}} for g, v in sorted(groups.items())},
          "groups_trainval": sorted({r["group"] for r in tv}), "groups_test": sorted({r["group"] for r in te}),
          "groups_val": sorted({r["group"] for r in val})}

    # --- boxovi
    st["boxes"] = {}
    tables = {"train": box_table(train), "val": box_table(val), "test": box_table(te), "trainval": box_table(tv)}
    for name, recs in [("train", train), ("val", val), ("test", te), ("trainval", tv)]:
        t = tables[name]
        per_img = [len(r["rows"]) for r in recs]
        st["boxes"][name] = {
            "images": len(recs), "boxes": len(t), "per_image": describe(per_img),
            "per_class": {NAMES[c]: int((t[:, 4] == c).sum()) for c in NAMES},
            "images_with_class": {NAMES[c]: int(sum(1 for r in recs if (r["rows"][:, 4] == c).any())) for c in NAMES},
            "distinct_classes_per_image": round(float(np.mean([len(set(r["rows"][:, 4])) for r in recs])), 2),
        }
    t = tables["trainval"]
    allb = np.vstack([tables["trainval"], tables["test"]])
    size = np.sqrt(allb[:, 5] * allb[:, 6])
    area_frac = allb[:, 5] * allb[:, 6] / IMG ** 2
    st["size_px"] = {"all": describe(size), "width": describe(allb[:, 5]), "height": describe(allb[:, 6]),
                     "area_fraction_percent": describe(area_frac * 100)}
    st["size_by_class"] = {}
    for c in NAMES:
        m = allb[:, 4] == c
        st["size_by_class"][NAMES[c]] = {
            "size": describe(size[m]), "w": round(float(allb[m, 5].mean()), 1), "h": round(float(allb[m, 6].mean()), 1),
            "aspect_w_over_h": round(float(np.median(allb[m, 5] / allb[m, 6])), 2)}
    st["size_mm"] = {"mean_side_mm": round(float(size.mean() / PX_PER_MM), 2), "image_side_mm": round(IMG / PX_PER_MM, 1)}
    st["boxes_under_1pct_of_image"] = round(float((area_frac < 0.01).mean() * 100), 1)
    st["boxes_under_32px"] = round(float((size < 32).mean() * 100), 1)
    # --- ispravnost anotacija, preklapanja, ivice
    out_of_image = (allb[:, 0] < 0) | (allb[:, 1] < 0) | (allb[:, 2] > IMG) | (allb[:, 3] > IMG)
    bad = int((out_of_image | (allb[:, 5] <= 0) | (allb[:, 6] <= 0)).sum())
    border = int(((allb[:, 0] <= 2) | (allb[:, 1] <= 2) | (allb[:, 2] >= IMG - 2) | (allb[:, 3] >= IMG - 2)).sum())
    overlaps = 0
    for r in tv + te:
        b = r["rows"][:, :4]
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                if min(b[i, 2], b[j, 2]) > max(b[i, 0], b[j, 0]) and min(b[i, 3], b[j, 3]) > max(b[i, 1], b[j, 1]):
                    overlaps += 1
    st["annotation_checks"] = {"invalid_boxes": bad, "boxes_touching_border": border, "overlapping_box_pairs": overlaps}

    # --- svojstva slika i para
    ps = [pair_stats(r) for r in tv + te]
    st["image"] = {"mode": sorted({p["mode"] for p in ps}), "sizes": sorted({str(p["size"]) for p in ps}),
                   "dark_fraction": describe([p["dark"] for p in ps]),
                   "pixels_within_10_of_black_or_white_percent": round(float(np.mean([p["extreme"] for p in ps]) * 100), 2)}
    outside = [p["diff_outside"] / p["diff"] for p in ps if p["diff"] > 0]
    st["difference"] = {
        "diff_pixels_per_image": describe([p["diff"] for p in ps]),
        "share_of_diff_pixels_outside_boxes": describe(outside),
        "overall_share_outside_boxes_percent": round(100 * sum(p["diff_outside"] for p in ps) / sum(p["diff"] for p in ps), 1),
    }
    structured = [p for p in ps if p["edges"] >= 300]
    nz = [p for p in structured if p["shift"] != (0, 0) and p["after"] < 0.8 * p["before"]]
    magn = [max(abs(p["shift"][0]), abs(p["shift"][1])) for p in nz]
    st["alignment"] = {
        "pairs_with_structure": len(structured), "pairs_without_structure": len(ps) - len(structured),
        "pairs_where_shift_reduces_difference_by_20pct": len(nz),
        "share_percent": round(100 * len(nz) / max(len(structured), 1), 1),
        "shift_magnitude_px": describe(magn) if magn else None,
        "median_diff_pixels_before_after_percent": [round(float(np.median([p["before"] for p in structured])) * 100, 3),
                                                      round(float(np.median([p["after"] for p in structured])) * 100, 3)],
        "shift_histogram_max_abs": dict(sorted(collections.Counter(magn).items())),
    }
    return st, {"train": train, "val": val, "test": te, "tv": tv, "tables": tables, "pairs": ps}


def make_figures(st, ctx, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = [NAMES[c] for c in NAMES]
    cols = ["#4a78b5", "#d98a2b"]

    # 1. raspodjela klasa
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(6)
    tr, te = st["boxes"]["trainval"]["per_class"], st["boxes"]["test"]["per_class"]
    ax.bar(x - 0.2, [tr[n] for n in names], 0.4, label="trening + validacija (1000 slika)", color=cols[0])
    ax.bar(x + 0.2, [te[n] for n in names], 0.4, label="test (500 slika)", color=cols[1])
    for i, n in enumerate(names):
        ax.text(i - 0.2, tr[n] + 15, str(tr[n]), ha="center", fontsize=8)
        ax.text(i + 0.2, te[n] + 15, str(te[n]), ha="center", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylabel("broj defekata")
    ax.set_title("Raspodjela klasa")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "01_raspodjela_klasa.png", dpi=130)
    plt.close(fig)

    # 2. defekata po slici
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for recs, lab, c in [(ctx["tv"], "trening + validacija", cols[0]), (ctx["test"], "test", cols[1])]:
        v = [len(r["rows"]) for r in recs]
        ax.hist(v, bins=np.arange(0.5, max(v) + 1.5), alpha=0.6, label=lab, color=c, density=True)
    ax.set_xlabel("broj defekata na slici")
    ax.set_ylabel("udio slika")
    ax.set_title("Broj defekata po slici")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "02_defekata_po_slici.png", dpi=130)
    plt.close(fig)

    # 3. veličine boxova po klasi
    allb = np.vstack([ctx["tables"]["trainval"], ctx["tables"]["test"]])
    size = np.sqrt(allb[:, 5] * allb[:, 6])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.boxplot([size[allb[:, 4] == c] for c in NAMES], tick_labels=names, showfliers=True, flierprops={"markersize": 2})
    ax.set_ylabel("veličina boxa, sqrt(širina·visina) [px]")
    ax.set_title("Veličina defekata po klasi (slika je 640 × 640 px)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "03_velicine_boxova.png", dpi=130)
    plt.close(fig)

    # 4. prostorni raspored
    cx, cy = (allb[:, 0] + allb[:, 2]) / 2, (allb[:, 1] + allb[:, 3]) / 2
    fig, ax = plt.subplots(figsize=(5, 4.6))
    h = ax.hist2d(cx, cy, bins=20, range=[[0, IMG], [0, IMG]], cmap="viridis")
    ax.invert_yaxis()
    ax.set_title("Položaj centara defekata na slici")
    fig.colorbar(h[3], ax=ax, label="broj defekata")
    fig.tight_layout()
    fig.savefig(out / "04_prostorni_raspored.png", dpi=130)
    plt.close(fig)

    # 5. primjeri po klasi: crop test | template
    rng = np.random.default_rng(0)
    pad = 14
    fig, axes = plt.subplots(6, 8, figsize=(12, 9))
    cache = {}
    for ci, c in enumerate(NAMES):
        cand = [(k, r) for k, rec in enumerate(ctx["tv"]) for r in rec["rows"] if r[4] == c]
        pick = rng.choice(len(cand), 4, replace=False)
        for j, pi in enumerate(pick):
            k, r = cand[pi]
            rec = ctx["tv"][k]
            if k not in cache:
                cache[k] = (np.array(Image.open(rec["test"]).convert("L")), np.array(Image.open(rec["temp"]).convert("L")))
            t, p = cache[k]
            x1, y1, x2, y2 = max(r[0] - pad, 0), max(r[1] - pad, 0), min(r[2] + pad, IMG), min(r[3] + pad, IMG)
            for q, arr in enumerate([t, p]):
                ax = axes[ci, 2 * j + q]
                ax.imshow(arr[y1:y2, x1:x2], cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                if ci == 0:
                    ax.set_title("test" if q == 0 else "template", fontsize=8)
            if j == 0:
                axes[ci, 0].set_ylabel(NAMES[c], fontsize=10)
    fig.suptitle("Primjeri defekata po klasi: isječak test slike i odgovarajući isječak template slike", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "05_primjeri_po_klasi.png", dpi=110)
    plt.close(fig)

    # 6. jedan par: test + boxovi | template | razlika
    rec = ctx["val"][0]
    t = np.array(Image.open(rec["test"]).convert("L"))
    p = np.array(Image.open(rec["temp"]).convert("L"))
    d = np.abs(t.astype(np.int16) - p.astype(np.int16)) > DIFF_THR
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ax[0].imshow(t, cmap="gray")
    for x1, y1, x2, y2, c in rec["rows"]:
        ax[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
        ax[0].text(x1, y1 - 3, NAMES[c], color="red", fontsize=7)
    ax[1].imshow(p, cmap="gray")
    ax[2].imshow(~d, cmap="gray")
    for a, ti in zip(ax, ["test slika s anotacijama", "template (ispravna ploča)", f"razlika |test − template| > {DIFF_THR}"]):
        a.set_title(ti, fontsize=10)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "06_primjer_para.png", dpi=110)
    plt.close(fig)

    # 7. grupe
    gs = st["groups"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    gl = list(gs)
    ax.bar(gl, [gs[g]["trainval"] for g in gl], label="trening + validacija", color=cols[0])
    ax.bar(gl, [gs[g]["test"] for g in gl], bottom=[gs[g]["trainval"] for g in gl], label="test", color=cols[1])
    ax.set_ylabel("broj parova slika")
    ax.set_title("Slike po grupama ploča")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "07_grupe.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    st, ctx = compute(a.root)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "eda_stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(st, ctx, a.out)
    print(json.dumps(st, indent=2, ensure_ascii=False))
    print(f"\nSlike i eda_stats.json zapisani u {a.out}/")


if __name__ == "__main__":
    main()
