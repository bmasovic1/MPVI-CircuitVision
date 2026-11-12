"""Provjera evaluacije na lažnim predikcijama kojima su rezultati poznati unaprijed.

Pokretanje (potreban je preuzet dataset):  python -m src.evaluation.check_fake
Ako sve prođe, ispiše "SVE PROVJERE PROŠLE". Pokrenuti nakon svake izmjene metrics.py.
"""
import tempfile
import time
from pathlib import Path

import numpy as np

from .analysis import analyze, draw_errors, match_detail, plot_confusion_matrix, recall_by_size
from .compare import append_result, comparison_table, measure_inference_time, read_results, result_row
from .metrics import (average_precision, collect_matches, evaluate, get_split_items, iou_matrix,
                      load_ground_truth, load_predictions, save_predictions, validate_predictions)
from .robustness import add_noise, blur, change_contrast, run_robustness, shift_gts, shift_image


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def from_gt(gts, box_fn=None, keep_fn=None, cls_fn=None, score=1.0):
    """Pravi predikcije iz ground trutha uz zadate izmjene (za lažne scenarije)."""
    preds = {}
    for img, (b, c) in gts.items():
        keep = np.ones(len(c), bool) if keep_fn is None else keep_fn(len(c))
        bb = b[keep].copy()
        cc = c[keep].copy()
        if box_fn is not None:
            bb = box_fn(bb)
        if cls_fn is not None:
            cc = cls_fn(cc)
        preds[img] = (bb, np.full(len(cc), score), cc)
    return preds


def main():
    gts = load_ground_truth("val")
    n_gt = sum(len(c) for _, c in gts.values())
    print(f"Validacijski skup: {len(gts)} slika, {n_gt} GT boxova\n")

    # 0. IoU na poznatim primjerima
    a = np.array([[0, 0, 10, 10]], float)
    assert close(iou_matrix(a, a)[0, 0], 1.0)
    assert close(iou_matrix(a, np.array([[20, 20, 30, 30]]))[0, 0], 0.0)
    assert close(iou_matrix(a, np.array([[5, 0, 15, 10]]))[0, 0], 50 / 150)  # presjek 50, unija 150
    print("OK  IoU: identični = 1, bez preklapanja = 0, pomak za pola širine = 1/3")

    # 1. Ručno izračunat AP: 2 GT; predikcije 0.9 (pogodak), 0.8 (promašaj), 0.7 (pogodak)
    g = {"x": (np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float), np.array([1, 1]))}
    p = {"x": (np.array([[0, 0, 10, 10], [100, 100, 110, 110], [50, 50, 60, 60]], float),
               np.array([0.9, 0.8, 0.7]), np.array([1, 1, 1]))}
    s, t, n = collect_matches(g, p, 0.5, 1)
    ap = average_precision(s, t, n)
    assert close(ap, 0.5 * 1.0 + 0.5 * (2 / 3)), ap
    print(f"OK  ručni primjer: AP = {ap:.4f} (očekivano 0.8333)")

    # 2. Savršene predikcije (tačne anotacije)
    r = evaluate(gts, from_gt(gts))
    assert close(r["mAP50"], 1) and close(r["mAP50_95"], 1) and close(r["mAP33"], 1)
    assert close(r["iou50"]["precision"], 1) and close(r["iou50"]["recall"], 1) and close(r["iou50"]["f1"], 1)
    print("OK  savršene predikcije: mAP = P = R = F1 = 1")

    # 3. Svaki drugi box izostavljen: P = 1, R = zadržani / ukupno
    kept = [0]

    def every_other(n):
        k = np.arange(n) % 2 == 0
        kept[0] += int(k.sum())
        return k

    r = evaluate(gts, from_gt(gts, keep_fn=every_other))
    assert close(r["iou50"]["precision"], 1.0) and close(r["iou50"]["recall"], kept[0] / n_gt)
    print(f"OK  polovina boxova izostavljena: P = 1, R = {r['iou50']['recall']:.4f} (= {kept[0]}/{n_gt})")

    # 4. Pomak boxova tako da je IoU = 0.4: pogodak pri pragu 0.33, promašaj pri 0.5
    def shift_iou04(b):
        b = b.copy()
        w = b[:, 2] - b[:, 0]
        s = w * (1 - 0.4) / (1 + 0.4)  # za isti oblik pomaknut samo po x: IoU = (w - s) / (w + s)
        b[:, 0] += s
        b[:, 2] += s
        return b

    r = evaluate(gts, from_gt(gts, box_fn=shift_iou04))
    assert close(r["mAP33"], 1) and close(r["mAP50"], 0) and close(r["mAP50_95"], 0)
    assert r["iou33"]["fn"] == 0 and r["iou50"]["tp"] == 0
    print("OK  pomak (IoU = 0.4): mAP@0.33 = 1, mAP@0.5 = 0")

    # 5. Pogrešna klasa: po klasama 0, a bez klase (cls = 0) 1
    r = evaluate(gts, from_gt(gts, cls_fn=lambda c: c % 6 + 1))
    assert close(r["mAP50"], 0) and r["iou50"]["tp"] == 0
    r = evaluate(gts, from_gt(gts, cls_fn=lambda c: np.zeros_like(c)))
    assert r["mode"].startswith("bez klase") and close(r["mAP50"], 1) and close(r["iou50"]["f1"], 1)
    print("OK  pogrešna klasa: mAP = 0 po klasama; cls = 0 daje način bez klase i mAP = 1")

    # 6. Duplikati s nižim scoreom: precision pri conf = 0.25 pada na 0.5, AP ostaje 1
    base = from_gt(gts, score=0.9)
    dup = {i: (np.concatenate([b, b]), np.concatenate([s, np.full(len(s), 0.5)]), np.concatenate([c, c]))
           for i, (b, s, c) in base.items()}
    r = evaluate(gts, dup)
    assert close(r["mAP50"], 1) and close(r["iou50"]["precision"], 0.5) and close(r["iou50"]["recall"], 1)
    print("OK  duplikati: jedan GT = najviše jedna predikcija (P = 0.5, R = 1, AP = 1)")

    # 7. Upis i čitanje JSON-a u formatu iz FORMAT.md daje isti rezultat
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "predictions_val.json"
        orig = from_gt(gts, box_fn=shift_iou04)
        save_predictions(orig, f)
        back = load_predictions(f)
        assert validate_predictions(gts, back) == []
        assert close(evaluate(gts, back)["mAP33"], evaluate(gts, orig)["mAP33"])
    print("OK  JSON upis/čitanje (FORMAT.md) i validacija predikcija")

    # 8. Validacija hvata greške
    bad = from_gt(gts)
    first = next(iter(bad))
    del bad[first]
    assert any("nema zapis" in w for w in validate_predictions(gts, bad))
    print("OK  validacija prijavljuje slike bez zapisa")

    # 9. Matrica konfuzije
    an = analyze(gts, from_gt(gts))
    cm = np.array(an["confusion_matrix"])
    per_class = [int(sum(1 for _, c in gts.values() for x in c if x == k)) for k in range(1, 7)]
    assert (np.diag(cm)[:6] == per_class).all() and cm.sum() == n_gt
    an = analyze(gts, from_gt(gts, cls_fn=lambda c: c % 6 + 1))
    cm = np.array(an["confusion_matrix"])
    assert all(cm[i, (i + 1) % 6] == per_class[i] for i in range(6)) and an["wrong_class"] == n_gt
    an = analyze(gts, from_gt(gts, keep_fn=every_other))
    cm = np.array(an["confusion_matrix"])
    assert an["correct"] == kept[0] // 2 and cm[:6, 6].sum() == n_gt - an["correct"]
    far, added = {}, 0
    for img, (b, c, ) in gts.items():
        box = np.array([[600.0, 600.0, 620.0, 620.0]])
        ok = iou_matrix(box, b).max(initial=0) == 0
        added += int(ok)
        far[img] = (np.vstack([b, box]) if ok else b, np.ones(len(c) + int(ok)),
                    np.concatenate([c, [1]]) if ok else c)
    an = analyze(gts, far)
    assert an["false_positives"] == added and np.array(an["confusion_matrix"])[6].sum() == added
    print("OK  matrica konfuzije: dijagonala, zamjene klasa, promašaji (kolona pozadina), lažne uzbune (red pozadina)")

    # 10. Recall po veličini defekta
    an = analyze(gts, from_gt(gts))
    sizes = recall_by_size(an["gt_rows"])
    assert sum(r["n"] for r in sizes) == n_gt and all(r["recall"] == 1.0 for r in sizes if r["n"])
    print("OK  recall po veličini: svi razredi veličine imaju 1.0 za savršene predikcije")

    # 11. Slike (matrica konfuzije i primjer grešaka)
    items = get_split_items("val")
    with tempfile.TemporaryDirectory() as d:
        plot_confusion_matrix(analyze(gts, from_gt(gts)), Path(d) / "cm.png")
        draw_errors(items[0], gts, from_gt(gts, cls_fn=lambda c: c % 6 + 1), Path(d) / "e.png")
        assert (Path(d) / "cm.png").stat().st_size > 1000 and (Path(d) / "e.png").stat().st_size > 1000
    print("OK  crtanje matrice konfuzije i slike s greškama")

    # 12. results.csv i tabela poređenja
    res = evaluate(gts, from_gt(gts))
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "results.csv"
        for seed, delta in [(0, 0.0), (1, 0.02)]:
            row = result_row("exp3", "yolo_template", seed, "val", res, 12.0, "test-hw")
            row["mAP50"] = round(0.90 + delta, 4)
            append_result(row, f)
        append_result(result_row("exp1", "classical", 0, "val", res, 3.0, "test-hw"), f)
        rows = read_results(f)
        assert len(rows) == 3 and rows[0]["mAP50"] == 0.9
        table = comparison_table(rows, "val")
        assert "0.910 ± 0.014" in table and "exp1" in table and "Upozorenje" not in table
    print("OK  results.csv (upis/čitanje) i tabela poređenja (prosjek ± odstupanje)")

    # 13. Mjerenje vremena
    t = measure_inference_time(lambda x: time.sleep(0.005), [0], warmup=2, runs=10)
    assert 4.5 <= t["ms_mean"] < 100
    print(f"OK  vrijeme inferencije: izmjereno {t['ms_mean']:.1f} ms za funkciju od ~5 ms")

    # 14. Poremećaji slike
    img = np.zeros((40, 40), np.uint8)
    img[10, 20] = 255
    moved = shift_image(img, 3, 2)
    assert moved[12, 23] == 255 and moved.sum() == 255
    assert shift_image(img, 0, 0).tolist() == img.tolist()
    sg = shift_gts({"a": (np.array([[10.0, 10, 20, 20], [636, 10, 640, 20]]), np.array([1, 2]))}, 5, 0)
    assert sg["a"][0].tolist() == [[15.0, 10, 25, 20]] and sg["a"][1].tolist() == [1]  # drugi box potpuno izlazi iz slike
    flat = np.full((64, 64), 128, np.uint8)
    noisy = add_noise(flat, 10, np.random.default_rng(0)).astype(float)
    assert abs(noisy.std() - 10) < 1.0 and abs(noisy.mean() - 128) < 1.0
    chess = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    assert blur(chess, 1.0).std() < chess.std() and change_contrast(chess, 1.0).tolist() == chess.tolist()
    print("OK  poremećaji: pomak slike i GT-a, šum (sigma), blur smanjuje kontrast, kontrast 1.0 = bez promjene")

    # 15. Cijeli tok robusnosti s "oracle" modelom koji vraća originalni GT i ne gleda sliku
    all_gts = load_ground_truth("val")

    def oracle(test_img, temp_img, iid):
        b, c = all_gts[iid]
        return b, np.ones(len(c)), c

    small = {k: v for k, v in __import__("src.evaluation.robustness", fromlist=["x"]).PERTURBATIONS.items()}
    small = {"noise": {**small["noise"], "levels": [10]}, "blur": {**small["blur"], "levels": [1.0]},
             "contrast": {**small["contrast"], "levels": [0.5]}, "shift": {**small["shift"], "levels": [2, 8]}}
    r = run_robustness(oracle, "fake", "oracle", "val", perturbations=small, limit=10)
    by = {(x["perturbation"], x["level"]): x for x in r}
    assert by[("clean", 0)]["mAP50"] == 1.0
    assert all(by[(k, lv)]["delta_mAP50"] == 0 for k, lv in [("noise", 10), ("blur", 1.0), ("contrast", 0.5)])
    assert by[("shift", 2)]["mAP50_95"] < 1.0 and by[("shift", 8)]["mAP50_95"] < by[("shift", 2)]["mAP50_95"]
    print("OK  robusnost: model koji ne gleda sliku ne mijenja se na šum/blur/kontrast, a pomak GT-a ga kažnjava")

    # 16. Indeksi predikcija ostaju originalni i kad conf filtrira dio predikcija (ispravka greške u crtanju)
    g1 = {"x": (np.array([[100.0, 100, 140, 140]]), np.array([1]))}
    p1 = {"x": (np.array([[300.0, 300, 340, 340], [100.0, 100, 140, 140]]), np.array([0.1, 0.9]), np.array([1, 1]))}
    _, pr = match_detail(g1, p1, 0.5, 0.25)
    assert pr[0]["idx"] == 1 and p1["x"][0][pr[0]["idx"]].tolist() == [100.0, 100, 140, 140]
    print("OK  indeksi predikcija nakon filtriranja po conf pokazuju na pravi box")

    # 17. Neispravne predikcije prekidaju evaluaciju; nedostajuće slike su samo upozorenje
    def raises(preds):
        try:
            evaluate(g1, preds)
        except ValueError:
            return True
        return False

    box = np.array([[100.0, 100, 140, 140]])
    assert raises({"x": (box, np.array([0.9]), np.array([7]))})                          # klasa 7
    assert raises({"x": (np.vstack([box, box]), np.array([0.9, 0.9]), np.array([0, 1]))})  # miješano 0 i 1-6
    assert raises({"x": (np.array([[140.0, 100, 100, 140]]), np.array([0.9]), np.array([1]))})  # x1 > x2
    assert raises({"x": (box, np.array([1.5]), np.array([1]))})                          # score > 1
    assert raises({"x": (np.array([[np.nan, 100, 140, 140]]), np.array([0.9]), np.array([1]))})  # NaN
    try:
        validate_predictions(g1, {"x": (box, np.array([0.9]), np.array([-1]))})
        raise AssertionError("validate_predictions je trebala prekinuti")
    except ValueError:
        pass
    two = {"x": g1["x"], "y": g1["x"]}
    assert any("nema zapis" in w for w in validate_predictions(two, {"x": (box, np.array([0.9]), np.array([1]))}))
    assert evaluate(g1, {"x": (box, np.array([0.9]), np.array([1]))})["mAP50"] == 1.0
    print("OK  neispravne predikcije (klasa 7, miješane klase, x1 > x2, score > 1, NaN) prekidaju evaluaciju")

    print("\nSVE PROVJERE PROŠLE")


if __name__ == "__main__":
    main()
