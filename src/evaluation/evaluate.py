"""Pokretanje evaluacije nad fajlom s predikcijama.

Primjer:
  python -m src.evaluation.evaluate --pred results/exp3_yolo/predictions_val.json --split val
"""
import argparse
import json

from .metrics import format_report, evaluate, load_ground_truth, load_predictions, validate_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions_<skup>.json (format: docs/FORMAT.md)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--root", default="data/DeepPCB")
    ap.add_argument("--conf", type=float, default=0.25, help="prag pouzdanosti za P/R/F1")
    ap.add_argument("--json-out", help="opcionalno: upiši rezultate u JSON")
    a = ap.parse_args()

    gts = load_ground_truth(a.split, a.root)
    preds = load_predictions(a.pred)
    try:
        for w in validate_predictions(gts, preds):
            print("UPOZORENJE:", w)
        res = evaluate(gts, preds, conf=a.conf)
    except ValueError as e:
        raise SystemExit(f"GREŠKA: {e}")
    print(format_report(res))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
