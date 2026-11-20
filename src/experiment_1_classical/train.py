"""Treniranje RF i SVM modela na kandidatima (lokalno i na Google Colabu).

Tok: slike -> kandidati -> karakteristike i oznake (build_dataset) -> cache na disku -> treniranje -> modeli.
  * Trenira se SAMO na `train` skupu; `val` služi za izbor hiperparametara i izvještaj, a `test` se ovdje ne dira.
  * Prije treniranja se provjerava da train i val ne dijele ploče (grupe), da nema curenja podataka.
  * Karakteristike se keširaju (npz) pod ključem koji zavisi od konfiguracije kandidata/karakteristika/podjele,
    pa promjena konfiguracije ne može iskoristiti zastarjeli cache. Na Colabu cache ide na Google Drive.
  * RF se trenira s više seedova (cfg.train.rf_seeds), SVM s cfg.train.svm_seeds.
  * Metrike u ovom modulu su na nivou KANDIDATA (tačnost, macro F1) i služe za izbor hiperparametara;
    detekcijske metrike (mAP, F1 po slici) računa evaluacija nad predikcijama (koraci 8-9).

Izlaz u results/exp1_<rf|svm>/: model_seed<N>/ (model.joblib + model.json), config.json (korištena konfiguracija,
uključujući izabrane hiperparametre) i train_summary.json.

Primjer (Colab ili lokalno, iz korijena repoa):
    python -m src.experiment_1_classical.train --model rf
    python -m src.experiment_1_classical.train --model svm --tune --cache-dir /content/drive/MyDrive/circuitvision/cache
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.data.deeppcb import NAMES

from .config import BACKGROUND, DEFECT_CLASSES, ExperimentConfig
from .dataset import Item, get_items, iter_pairs
from .features import extract_features, feature_names
from .labeling import label_pair
from .models import MODEL_NAMES, build_model, save_model

ALL_CLASSES = (BACKGROUND, *DEFECT_CLASSES)
CLASS_NAMES = {BACKGROUND: "pozadina", **NAMES}


@dataclass(frozen=True)
class CandidateDataset:
    """Karakteristike i oznake svih kandidata jednog skupa; image_ids i groups služe za analizu i provjeru curenja."""
    X: np.ndarray  # (N, D) float32
    y: np.ndarray  # (N,) int, 0 = pozadina, 1-6 = tip defekta
    image_ids: np.ndarray  # (N,) str
    groups: np.ndarray  # (N,) str
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)


# ----------------------------------------------------------------------------- izgradnja i cache skupa

def build_dataset(items: list[Item], cfg: ExperimentConfig | None = None) -> CandidateDataset:
    """Izdvaja kandidate, njihove karakteristike i oznake za sve parove (slike bez kandidata se preskaču)."""
    cfg = cfg or ExperimentConfig()
    names = feature_names(cfg.features)
    xs, ys, ids, groups = [], [], [], []
    for pair in iter_pairs(items):
        cand, labels = label_pair(pair, cfg)
        if len(cand) == 0:
            continue
        xs.append(extract_features(pair.test, pair.temp, cand, cfg.features))
        ys.append(labels)
        ids += [pair.image_id] * len(cand)
        groups += [pair.group] * len(cand)
    X = np.vstack(xs) if xs else np.zeros((0, len(names)), dtype=np.float32)
    y = np.concatenate(ys) if ys else np.zeros(0, dtype=int)
    return CandidateDataset(X, y, np.array(ids, dtype=str), np.array(groups, dtype=str), names)


def dataset_key(split: str, cfg: ExperimentConfig) -> str:
    """Kratki ključ cachea: zavisi od skupa i od svega što utiče na kandidate, oznake i karakteristike."""
    payload = {"split": split, "split_cfg": asdict(cfg.split), "candidates": asdict(cfg.candidates), "features": asdict(cfg.features)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def save_dataset(ds: CandidateDataset, path: str | Path) -> None:
    """Upisuje skup u npz (atomarno: prvo privremeni fajl, da prekid na Colabu ne ostavi polovičan cache)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, X=ds.X, y=ds.y, image_ids=ds.image_ids, groups=ds.groups, feature_names=np.array(ds.feature_names))
    os.replace(tmp, p)


def load_dataset(path: str | Path) -> CandidateDataset:
    """Učitava skup iz npz; ValueError ako nedostaju polja ili se oblici ne poklapaju."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ("X", "y", "image_ids", "groups", "feature_names") if k not in z.files]
        if missing:
            raise ValueError(f"{path}: nedostaju polja {missing}")
        ds = CandidateDataset(z["X"], z["y"], z["image_ids"], z["groups"], [str(n) for n in z["feature_names"]])
    if not (len(ds.X) == len(ds.y) == len(ds.image_ids) == len(ds.groups)) or ds.X.shape[1] != len(ds.feature_names):
        raise ValueError(f"{path}: oblici polja se ne poklapaju")
    return ds


def load_or_build(split: str, cfg: ExperimentConfig, cache_dir: str | Path | None = None) -> tuple[CandidateDataset, bool]:
    """Skup iz cachea ako postoji i ispravan je, inače se gradi i upisuje. Vraća (skup, iz_cachea)."""
    cache = Path(cache_dir) if cache_dir else Path(cfg.results_root) / "exp1_cache"
    path = cache / f"features_{split}_{dataset_key(split, cfg)}.npz"
    if path.is_file():
        try:
            ds = load_dataset(path)
            if ds.feature_names == feature_names(cfg.features):
                return ds, True
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
            warnings.warn(f"cache {path} je neispravan ({e}); gradim ponovo", UserWarning, stacklevel=2)
    ds = build_dataset(get_items(split, cfg), cfg)
    save_dataset(ds, path)
    return ds, False


def ensure_disjoint_groups(train: CandidateDataset, val: CandidateDataset) -> None:
    """Sprečava curenje podataka: ista ploča ne smije imati kandidate i u train i u val skupu."""
    overlap = sorted(set(train.groups.tolist()) & set(val.groups.tolist()))
    if overlap:
        raise ValueError(f"grupe {overlap} su i u train i u val skupu (curenje podataka)")


# ----------------------------------------------------------------------------- treniranje i metrike

def _check_dataset(ds: CandidateDataset) -> None:
    if len(ds) == 0:
        raise ValueError("skup kandidata je prazan")
    if ds.X.ndim != 2 or len(ds.X) != len(ds.y):
        raise ValueError(f"X mora biti 2D i imati isti broj redova kao y (dobijeno {ds.X.shape}, {ds.y.shape})")
    if not np.isfinite(ds.X).all():
        raise ValueError("X sadrži NaN ili beskonačne vrijednosti")
    if not np.isin(ds.y, ALL_CLASSES).all():
        raise ValueError(f"oznake moraju biti u {ALL_CLASSES}, dobijeno {sorted(set(ds.y.tolist()))}")


def train_model(name: str, cfg: ExperimentConfig, seed: int, train: CandidateDataset):
    """Trenira model ("rf" ili "svm") na train skupu. Skup mora imati pozadinu i bar jedan tip defekta."""
    _check_dataset(train)
    if BACKGROUND not in train.y or not (train.y > 0).any():
        raise ValueError("train skup mora sadržavati i kandidate 'pozadina' i bar jedan tip defekta")
    return build_model(name, cfg, seed).fit(train.X, train.y)


def candidate_metrics(model: Any, ds: CandidateDataset) -> dict[str, Any]:
    """Metrike na nivou kandidata: tačnost, macro F1 (samo klase prisutne u ds), defekt/pozadina i F1 po klasi."""
    _check_dataset(ds)
    pred = model.predict(ds.X)
    f1 = f1_score(ds.y, pred, labels=list(ALL_CLASSES), average=None, zero_division=0)
    support = np.bincount(ds.y, minlength=len(ALL_CLASSES))
    present = [f for f, s in zip(f1, support) if s > 0]
    return {"n": len(ds), "accuracy": float(accuracy_score(ds.y, pred)), "macro_f1": float(np.mean(present)),
            "defect_vs_background_accuracy": float(accuracy_score(ds.y > 0, pred > 0)),
            "f1": {CLASS_NAMES[c]: (float(f) if s > 0 else None) for c, f, s in zip(ALL_CLASSES, f1, support)}}


def grid_configs(name: str, cfg: ExperimentConfig) -> list[tuple[dict[str, Any], ExperimentConfig]]:
    """Sve kombinacije hiperparametara iz cfg.tune kao (parametri, konfiguracija)."""
    if name not in MODEL_NAMES:
        raise ValueError(f"nepoznat model {name!r}; dozvoljeni: {MODEL_NAMES}")
    t = cfg.tune
    if name == "rf":
        grid = {"min_samples_leaf": t.rf_min_samples_leaf, "max_features": t.rf_max_features}
        make = lambda p: replace(cfg, random_forest=replace(cfg.random_forest, **p))  # noqa: E731
    else:
        grid = {"C": t.svm_C, "gamma": t.svm_gamma}
        make = lambda p: replace(cfg, svm=replace(cfg.svm, **p))  # noqa: E731
    return [(p, make(p)) for p in (dict(zip(grid, vals)) for vals in itertools.product(*grid.values()))]


def tune_model(name: str, cfg: ExperimentConfig, train: CandidateDataset, val: CandidateDataset,
               seed: int = 0) -> tuple[ExperimentConfig, list[dict[str, Any]]]:
    """Bira hiperparametre iz cfg.tune: trenira na train, ocjenjuje macro F1 na val. Vraća (najbolja konfiguracija, tabela)."""
    rows = []
    for params, candidate_cfg in grid_configs(name, cfg):
        t0 = time.perf_counter()
        model = train_model(name, candidate_cfg, seed, train)
        fit_seconds = time.perf_counter() - t0
        m = candidate_metrics(model, val)
        rows.append({"params": params, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "fit_seconds": fit_seconds})
        print(f"  tuning {name} {params}: val macro F1 {m['macro_f1']:.4f}, tačnost {m['accuracy']:.4f} ({fit_seconds:.0f} s)")
    best = max(range(len(rows)), key=lambda i: rows[i]["macro_f1"])  # prvi u slučaju izjednačenja
    return grid_configs(name, cfg)[best][1], rows


def run_training(name: str, cfg: ExperimentConfig, train: CandidateDataset, val: CandidateDataset,
                 seeds: tuple[int, ...] | None = None, tuning: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Trenira model za svaki seed, sprema modele, config.json i train_summary.json u results/exp1_<name>/."""
    ensure_disjoint_groups(train, val)
    seeds = seeds or (cfg.train.rf_seeds if name == "rf" else cfg.train.svm_seeds)
    out = cfg.results_dir(name)
    runs = []
    for seed in seeds:
        t0 = time.perf_counter()
        model = train_model(name, cfg, seed, train)
        fit_seconds = time.perf_counter() - t0
        metrics = candidate_metrics(model, val)
        save_model(model, out / f"model_seed{seed}", name, seed, cfg, train.feature_names)
        runs.append({"seed": int(seed), "fit_seconds": fit_seconds, "val": metrics})
        print(f"  {name} seed {seed}: val macro F1 {metrics['macro_f1']:.4f}, tačnost {metrics['accuracy']:.4f} ({fit_seconds:.0f} s)")
    f1s = [r["val"]["macro_f1"] for r in runs]
    summary = {"model": name, "n_train": len(train), "n_val": len(val), "feature_dim": int(train.X.shape[1]),
               "train_class_counts": {CLASS_NAMES[c]: int((train.y == c).sum()) for c in ALL_CLASSES},
               "runs": runs, "val_macro_f1_mean": float(np.mean(f1s)), "val_macro_f1_std": float(np.std(f1s)),
               "tuning": tuning}
    cfg.save(out / "config.json")
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="treniranje RF/SVM klasifikatora kandidata (Eksperiment 1)")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--config", help="JSON konfiguracija (npr. results/exp1_rf/config.json iz prethodnog runa)")
    ap.add_argument("--root", help="folder s DeepPCB-om (inače iz konfiguracije)")
    ap.add_argument("--results-root", help="folder za rezultate (inače iz konfiguracije), npr. folder na Google Drive-u")
    ap.add_argument("--cache-dir", help="folder za cache karakteristika (inače <results-root>/exp1_cache)")
    ap.add_argument("--tune", action="store_true", help="izaberi hiperparametre iz cfg.tune na val skupu (sporije)")
    ap.add_argument("--seeds", type=int, nargs="+", help="seedovi (inače iz konfiguracije)")
    a = ap.parse_args(argv)

    cfg = ExperimentConfig.load(a.config) if a.config else ExperimentConfig()
    cfg = replace(cfg, data_root=a.root or cfg.data_root, results_root=a.results_root or cfg.results_root)
    seeds = tuple(a.seeds) if a.seeds else None

    sets = {}
    for split in ("train", "val"):  # test se namjerno ne učitava
        t0 = time.perf_counter()
        sets[split], cached = load_or_build(split, cfg, a.cache_dir)
        print(f"{split}: {len(sets[split])} kandidata, {sets[split].X.shape[1]} karakteristika "
              f"({'iz cache-a' if cached else f'izgrađeno za {time.perf_counter() - t0:.0f} s'})")
    ensure_disjoint_groups(sets["train"], sets["val"])

    tuning = None
    if a.tune:
        print(f"izbor hiperparametara za {a.model} na val skupu:")
        cfg, tuning = tune_model(a.model, cfg, sets["train"], sets["val"], seed=(seeds or (0,))[0])
    print(f"treniranje {a.model}:")
    summary = run_training(a.model, cfg, sets["train"], sets["val"], seeds, tuning)
    print(f"val macro F1: {summary['val_macro_f1_mean']:.4f} +- {summary['val_macro_f1_std']:.4f}; rezultati u {cfg.results_dir(a.model)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
