"""Testovi RF i SVM modela, čuvanja i učitavanja (male sintetičke podloge, bez dataseta)."""
import json

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.experiment_1_classical.config import ExperimentConfig
from src.experiment_1_classical.models import (META_FILE, MODEL_FILE, MODEL_NAMES, build_model, load_model,
                                               save_model)

N_FEATURES = 6
NAMES = [f"f{i}" for i in range(N_FEATURES)]


@pytest.fixture(scope="module")
def data():
    """7 klasa (0 = pozadina, 1-6 defekti), 25 primjera po klasi, razdvojeni Gaussovi klasteri."""
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 6, size=(7, N_FEATURES))
    y = np.repeat(np.arange(7), 25)
    X = centers[y] + rng.normal(0, 1, size=(len(y), N_FEATURES))
    return X, y


@pytest.fixture(scope="module")
def cfg():
    return ExperimentConfig.from_dict({"random_forest": {"n_estimators": 30, "n_jobs": 1}})


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fit_predict_proba_interface(name, cfg, data):
    X, y = data
    model = build_model(name, cfg, seed=0).fit(X, y)
    proba = model.predict_proba(X)
    assert model.classes_.tolist() == list(range(7)) and proba.shape == (len(X), 7)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (model.predict(X) == y).mean() > 0.95


def test_hyperparameters_come_from_config():
    cfg = ExperimentConfig.from_dict({"random_forest": {"n_estimators": 17, "min_samples_leaf": 3, "n_jobs": 1},
                                      "svm": {"C": 2.5, "kernel": "linear"}})
    rf = build_model("rf", cfg, seed=4).named_steps["clf"]
    assert (rf.n_estimators, rf.min_samples_leaf, rf.class_weight, rf.random_state) == (17, 3, "balanced", 4)
    svm = build_model("svm", cfg, seed=4).named_steps["clf"]
    assert (svm.C, svm.kernel, svm.class_weight, svm.probability, svm.random_state) == (2.5, "linear", "balanced", True, 4)


def test_svm_has_scaler_and_rf_does_not():
    assert list(build_model("svm").named_steps) == ["scaler", "clf"]
    assert list(build_model("rf").named_steps) == ["clf"]


def test_svm_is_invariant_to_feature_scale(cfg, data):
    """Skaliranje je u pipelineu: množenje jedne karakteristike velikim brojem ne mijenja SVM predikcije."""
    X, y = data
    Xs = X.copy()
    Xs[:, 0] *= 1e4
    a = build_model("svm", cfg).fit(X, y).predict_proba(X)
    b = build_model("svm", cfg).fit(Xs, y).predict_proba(Xs)
    assert np.allclose(a, b, atol=1e-6)


def test_scaler_is_fitted_on_training_data_only(cfg, data):
    """Nema curenja: predikcija na drugim podacima ne mijenja parametre scalera."""
    X, y = data
    model = build_model("svm", cfg).fit(X, y)
    mean_before = model.named_steps["scaler"].mean_.copy()
    assert np.allclose(mean_before, X.mean(axis=0))
    model.predict_proba(X + 100.0)
    assert np.array_equal(model.named_steps["scaler"].mean_, mean_before)


def test_same_seed_is_reproducible_and_seeds_differ(cfg, data):
    X, y = data
    X_noisy = X + np.random.default_rng(1).normal(0, 4, X.shape)  # težak problem da se seedovi vide
    p0a = build_model("rf", cfg, seed=0).fit(X_noisy, y).predict_proba(X_noisy)
    p0b = build_model("rf", cfg, seed=0).fit(X_noisy, y).predict_proba(X_noisy)
    p1 = build_model("rf", cfg, seed=1).fit(X_noisy, y).predict_proba(X_noisy)
    assert np.array_equal(p0a, p0b) and not np.array_equal(p0a, p1)


def test_invalid_name_and_seed():
    with pytest.raises(ValueError, match="nepoznat model"):
        build_model("knn")
    with pytest.raises(ValueError, match="seed"):
        build_model("rf", seed=1.5)
    with pytest.raises(ValueError, match="seed"):
        build_model("rf", seed=True)


# ----------------------------------------------------------------------------- čuvanje i učitavanje

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_save_load_roundtrip(name, cfg, data, tmp_path):
    X, y = data
    model = build_model(name, cfg, seed=2).fit(X, y)
    out = save_model(model, tmp_path / "a" / "b", name, 2, cfg, NAMES)  # folder se kreira
    assert (out / MODEL_FILE).is_file() and (out / META_FILE).is_file()
    loaded, meta = load_model(out)
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))
    assert meta["model"] == name and meta["seed"] == 2 and meta["classes"] == list(range(7))
    assert meta["n_features"] == N_FEATURES and meta["feature_names"] == NAMES
    assert ExperimentConfig.from_dict(meta["config"]) == cfg  # tačna konfiguracija se može vratiti
    assert {"python", "scikit-learn", "numpy", "opencv", "joblib"} <= set(meta["versions"])


def test_save_rejects_unfitted_model_and_bad_inputs(cfg, data, tmp_path):
    X, y = data
    with pytest.raises(NotFittedError):
        save_model(build_model("rf", cfg), tmp_path, "rf", 0, cfg, NAMES)
    model = build_model("rf", cfg).fit(X, y)
    with pytest.raises(ValueError, match="naziva"):
        save_model(model, tmp_path, "rf", 0, cfg, NAMES[:-1])
    with pytest.raises(ValueError, match="nepoznat model"):
        save_model(model, tmp_path, "knn", 0, cfg, NAMES)


def test_load_missing_or_broken_files(cfg, data, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nema")
    X, y = data
    out = save_model(build_model("rf", cfg).fit(X, y), tmp_path / "m", "rf", 0, cfg, NAMES)
    (out / META_FILE).write_text("{ nije json")
    with pytest.raises(ValueError, match="neispravan"):
        load_model(out)
    (out / META_FILE).write_text(json.dumps({"model": "rf"}))
    with pytest.raises(ValueError, match="nema ključeve"):
        load_model(out)


def test_load_detects_mismatched_metadata(cfg, data, tmp_path):
    X, y = data
    out = save_model(build_model("rf", cfg).fit(X, y), tmp_path / "m", "rf", 0, cfg, NAMES)
    meta = json.loads((out / META_FILE).read_text())
    meta["n_features"] = 99
    (out / META_FILE).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="ne poklapaju"):
        load_model(out)


def test_load_warns_on_sklearn_version_mismatch(cfg, data, tmp_path):
    X, y = data
    out = save_model(build_model("rf", cfg).fit(X, y), tmp_path / "m", "rf", 0, cfg, NAMES)
    meta = json.loads((out / META_FILE).read_text())
    meta["versions"]["scikit-learn"] = "0.0.1"
    (out / META_FILE).write_text(json.dumps(meta))
    with pytest.warns(UserWarning, match="scikit-learn 0.0.1"):
        load_model(out)
