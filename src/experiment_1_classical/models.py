"""Random Forest i SVM klasifikatori kandidata: jedna tvornica, čuvanje i učitavanje modela.

Oba algoritma su sklearn `Pipeline` istog sučelja (fit, predict_proba, classes_), pa ostatak koda (treniranje,
detektor, evaluacija) ne zna koji algoritam koristi:
  rf   RandomForestClassifier (ne treba skaliranje)
  svm  StandardScaler + SVC; scaler se fituje samo na podacima za treniranje (nema curenja podataka)

Klase su 0 = pozadina i 1-6 = tipovi defekata (7-klasni model). Hiperparametri dolaze iz config.ExperimentConfig.

Čuvanje (u folder): model.joblib + model.json s metapodacima (algoritam, seed, klase, nazivi karakteristika,
cijela konfiguracija, verzije biblioteka). Model istreniran na Colabu učitava se lokalno preko load_model.
Napomena: joblib je pickle, pa se učitavaju samo vlastiti modeli, nikad tuđi fajlovi.

Primjer:
    model = build_model("svm", cfg, seed=0).fit(X_train, y_train)
    save_model(model, "results/exp1_svm/model_seed0", "svm", 0, cfg, feature_names(cfg.features))
    model, meta = load_model("results/exp1_svm/model_seed0")
"""
from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from .config import ExperimentConfig

MODEL_NAMES = ("rf", "svm")
MODEL_FILE = "model.joblib"
META_FILE = "model.json"
_META_KEYS = ("model", "seed", "classes", "n_features", "feature_names", "config", "versions")


def build_model(name: str, cfg: ExperimentConfig | None = None, seed: int = 0) -> Pipeline:
    """Neistreniran model ("rf" ili "svm") s hiperparametrima iz konfiguracije i zadanim seedom."""
    cfg = cfg or ExperimentConfig()
    if name not in MODEL_NAMES:
        raise ValueError(f"nepoznat model {name!r}; dozvoljeni: {MODEL_NAMES}")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError(f"seed mora biti cijeli broj, dobijeno {seed!r}")
    if name == "rf":
        c = cfg.random_forest
        clf = RandomForestClassifier(n_estimators=c.n_estimators, max_depth=c.max_depth, min_samples_leaf=c.min_samples_leaf,
                                     max_features=c.max_features, class_weight=c.class_weight, n_jobs=c.n_jobs,
                                     random_state=int(seed))
        return Pipeline([("clf", clf)])
    c = cfg.svm
    # NAPOMENA: scikit-learn >= 1.9 javlja FutureWarning za SVC(probability=True) (uklanja se u 1.11). Do tada radi;
    # zamjena je CalibratedClassifierCV(SVC(), ensemble=False). Verzija biblioteke se bilježi u model.json.
    clf = SVC(kernel=c.kernel, C=c.C, gamma=c.gamma, class_weight=c.class_weight, probability=c.probability,
              cache_size=1000, random_state=int(seed))  # random_state određuje interni CV za vjerovatnoće (Platt)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _versions() -> dict[str, str]:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__, "joblib": joblib.__version__}


def save_model(model: Pipeline, directory: str | Path, name: str, seed: int, cfg: ExperimentConfig,
               feature_names: list[str]) -> Path:
    """Upisuje istrenirani model i metapodatke u `directory` (kreira se). Vraća putanju foldera."""
    check_is_fitted(model)  # NotFittedError (podklasa ValueError) ako model nije istreniran
    if name not in MODEL_NAMES:
        raise ValueError(f"nepoznat model {name!r}; dozvoljeni: {MODEL_NAMES}")
    n_features = int(model.n_features_in_)
    if len(feature_names) != n_features:
        raise ValueError(f"feature_names ima {len(feature_names)} naziva, a model {n_features} karakteristika")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / MODEL_FILE, compress=3)
    meta = {"model": name, "seed": int(seed), "classes": [int(c) for c in model.classes_], "n_features": n_features,
            "feature_names": list(feature_names), "config": cfg.to_dict(), "versions": _versions()}
    (out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_model(directory: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    """Učitava model i metapodatke iz foldera koji je napravio save_model; provjerava da se poklapaju."""
    d = Path(directory)
    for f in (MODEL_FILE, META_FILE):
        if not (d / f).is_file():
            raise FileNotFoundError(f"nedostaje {d / f}")
    try:
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"neispravan {d / META_FILE}: {e}") from e
    missing = [k for k in _META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"{d / META_FILE} nema ključeve {missing}")
    model = joblib.load(d / MODEL_FILE)
    if model.n_features_in_ != meta["n_features"] or [int(c) for c in model.classes_] != meta["classes"]:
        raise ValueError(f"metapodaci u {d} se ne poklapaju s modelom (broj karakteristika ili klase)")
    saved = meta["versions"].get("scikit-learn")
    if saved != sklearn.__version__:
        warnings.warn(f"model je treniran s scikit-learn {saved}, a učitava se s {sklearn.__version__}; "
                      "rezultati se mogu razlikovati", UserWarning, stacklevel=2)
    return model, meta
