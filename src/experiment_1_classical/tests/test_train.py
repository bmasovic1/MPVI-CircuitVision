"""Testovi treniranja: skup kandidata, cache, treniranje, metrike, tuning i CLI (sintetički podaci, bez dataseta)."""
import json
import warnings

import numpy as np
import pytest
from PIL import Image

from src.experiment_1_classical import train as tr
from src.experiment_1_classical.config import ExperimentConfig
from src.experiment_1_classical.dataset import get_items
from src.experiment_1_classical.models import load_model

from .synthetic import make_dataset

TRAINVAL = {"group00041": 3, "group13000": 3, "group20085": 3, "group44000": 3, "group50600": 3}
TEST = {"group12100": 2}
FAST = {"random_forest": {"n_estimators": 10, "n_jobs": 1}}


def fake_root(path):
    """Dataset s po 2 kandidata po slici: mrlja s GT-om (klasa 5) i mrlja bez GT-a (pozadina)."""
    return make_dataset(path, TRAINVAL, TEST, gt=(290, 290, 340, 340, 5), spurious=True)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return fake_root(tmp_path_factory.mktemp("DeepPCB"))


@pytest.fixture(scope="module")
def cfg(root):
    return ExperimentConfig.from_dict({**FAST, "data_root": str(root)})


def synthetic_ds(classes=range(7), n_per_class=20, seed=0, group="g1", noise=1.0):
    """Gaussovi klasteri (isti centri za sve seedove, seed mijenja samo šum); klasa 0 je 'pozadina'."""
    centers = np.random.default_rng(123).normal(0, 6, size=(7, 5))
    rng = np.random.default_rng(seed)
    classes = list(classes)
    y = np.repeat(classes, n_per_class)
    X = (centers[y] + rng.normal(0, noise, size=(len(y), 5))).astype(np.float32)
    return tr.CandidateDataset(X, y, np.array([f"i{k}" for k in range(len(y))]), np.array([group] * len(y)),
                               [f"f{i}" for i in range(5)])


# ----------------------------------------------------------------------------- build_dataset

def test_build_dataset_labels_and_metadata(cfg):
    items = get_items("train", cfg)
    ds = tr.build_dataset(items, cfg)
    assert len(ds) == 2 * len(items) and ds.X.shape == (len(ds), len(ds.feature_names)) and ds.X.dtype == np.float32
    assert sorted(set(ds.y.tolist())) == [0, 5] and (ds.y == 5).sum() == (ds.y == 0).sum() == len(items)
    assert set(ds.groups.tolist()) == {i["group"] for i in items} and len(set(ds.image_ids.tolist())) == len(items)


def test_build_dataset_skips_images_without_candidates(tmp_path):
    cfg = ExperimentConfig(data_root=str(fake_root(tmp_path)))
    items = get_items("train", cfg)
    Image.open(items[0]["temp"]).save(items[0]["test"])  # test = template: nema razlike
    ds = tr.build_dataset(items, cfg)
    assert len(ds) == 2 * (len(items) - 1) and items[0]["ann"].stem not in set(ds.image_ids.tolist())


def test_build_dataset_with_no_candidates_is_empty(tmp_path):
    cfg = ExperimentConfig(data_root=str(fake_root(tmp_path)))
    items = get_items("test", cfg)
    for it in items:
        Image.open(it["temp"]).save(it["test"])
    ds = tr.build_dataset(items, cfg)
    assert len(ds) == 0 and ds.X.shape == (0, len(ds.feature_names))


# ----------------------------------------------------------------------------- cache

def test_dataset_save_load_roundtrip(tmp_path):
    ds = synthetic_ds()
    tr.save_dataset(ds, tmp_path / "x" / "d.npz")
    back = tr.load_dataset(tmp_path / "x" / "d.npz")
    assert np.array_equal(back.X, ds.X) and np.array_equal(back.y, ds.y) and back.feature_names == ds.feature_names
    assert np.array_equal(back.image_ids, ds.image_ids) and np.array_equal(back.groups, ds.groups)
    assert not list((tmp_path / "x").glob("*.tmp.npz"))  # privremeni fajl je preimenovan


def test_load_dataset_rejects_incomplete_file(tmp_path):
    np.savez(tmp_path / "bad.npz", X=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="nedostaju"):
        tr.load_dataset(tmp_path / "bad.npz")


def test_dataset_key_depends_on_what_matters():
    base = ExperimentConfig()
    k = tr.dataset_key("train", base)
    assert k == tr.dataset_key("train", ExperimentConfig())  # stabilan
    assert k != tr.dataset_key("val", base)
    assert k != tr.dataset_key("train", ExperimentConfig.from_dict({"candidates": {"box_padding": 8}}))
    assert k != tr.dataset_key("train", ExperimentConfig.from_dict({"features": {"groups": ["geometry"]}}))
    assert k != tr.dataset_key("train", ExperimentConfig.from_dict({"split": {"seed": 7}}))
    assert k == tr.dataset_key("train", ExperimentConfig.from_dict({"random_forest": {"n_estimators": 5}}))  # modeli ne utiču


def test_load_or_build_uses_cache_second_time(cfg, tmp_path, monkeypatch):
    first, cached1 = tr.load_or_build("val", cfg, tmp_path)
    assert not cached1
    monkeypatch.setattr(tr, "get_items", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache nije iskorišten")))
    second, cached2 = tr.load_or_build("val", cfg, tmp_path)
    assert cached2 and np.array_equal(first.X, second.X)


def test_load_or_build_rebuilds_corrupt_cache(cfg, tmp_path):
    tr.load_or_build("val", cfg, tmp_path)
    cache_file = next(tmp_path.glob("features_val_*.npz"))
    cache_file.write_bytes(b"ovo nije npz")
    with pytest.warns(UserWarning, match="neispravan"):
        ds, cached = tr.load_or_build("val", cfg, tmp_path)
    assert not cached and len(ds) > 0 and tr.load_dataset(cache_file).X.shape == ds.X.shape


def test_default_cache_dir_is_under_results_root(cfg, tmp_path):
    c = ExperimentConfig.from_dict({"data_root": cfg.data_root, "results_root": str(tmp_path / "res")})
    tr.load_or_build("val", c)
    assert list((tmp_path / "res" / "exp1_cache").glob("features_val_*.npz"))


def test_ensure_disjoint_groups():
    a, b = synthetic_ds(group="g1"), synthetic_ds(group="g2")
    tr.ensure_disjoint_groups(a, b)
    with pytest.raises(ValueError, match="curenje podataka"):
        tr.ensure_disjoint_groups(a, synthetic_ds(group="g1"))


# ----------------------------------------------------------------------------- treniranje i metrike

def test_train_model_fits_and_predicts():
    ds = synthetic_ds()
    model = tr.train_model("rf", ExperimentConfig.from_dict(FAST), 0, ds)
    assert model.classes_.tolist() == list(range(7)) and (model.predict(ds.X) == ds.y).mean() > 0.95


@pytest.mark.parametrize("make_bad, message", [
    (lambda d: tr.CandidateDataset(d.X[:0], d.y[:0], d.image_ids[:0], d.groups[:0], d.feature_names), "prazan"),
    (lambda d: tr.CandidateDataset(np.where(np.arange(len(d.X))[:, None] == 0, np.nan, d.X), d.y, d.image_ids, d.groups, d.feature_names), "NaN"),
    (lambda d: tr.CandidateDataset(d.X, np.where(d.y == 6, 9, d.y), d.image_ids, d.groups, d.feature_names), "oznake"),
    (lambda d: tr.CandidateDataset(d.X[:-1], d.y, d.image_ids, d.groups, d.feature_names), "isti broj redova"),
    (lambda d: synthetic_ds(classes=[1, 2]), "pozadina"),  # nema pozadine
    (lambda d: synthetic_ds(classes=[0]), "pozadina"),  # nema defekata
])
def test_train_model_rejects_unusable_data(make_bad, message):
    with pytest.raises(ValueError, match=message):
        tr.train_model("rf", ExperimentConfig.from_dict(FAST), 0, make_bad(synthetic_ds()))


class Const:
    """Lažni model koji uvijek predviđa zadane oznake."""
    def __init__(self, pred):
        self.pred = np.asarray(pred)

    def predict(self, X):
        return self.pred


def test_candidate_metrics_exact_values():
    ds = tr.CandidateDataset(np.zeros((4, 2), np.float32), np.array([0, 0, 5, 5]), np.array(list("abcd")), np.array(["g"] * 4), ["a", "b"])
    m = tr.candidate_metrics(Const([0, 5, 5, 5]), ds)  # jedan pogrešan: pozadina predviđena kao defekt
    assert m["n"] == 4 and m["accuracy"] == 0.75 and m["defect_vs_background_accuracy"] == 0.75
    assert m["f1"]["pozadina"] == pytest.approx(2 / 3) and m["f1"]["copper"] == pytest.approx(0.8)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)  # prosjek samo klasa prisutnih u skupu
    assert m["f1"]["open"] is None and "short" in m["f1"]  # klase bez primjera nisu 0, nego None


def test_candidate_metrics_defect_type_error_keeps_defect_vs_background():
    ds = tr.CandidateDataset(np.zeros((2, 1), np.float32), np.array([2, 3]), np.array(["a", "b"]), np.array(["g", "g"]), ["a"])
    m = tr.candidate_metrics(Const([3, 3]), ds)
    assert m["accuracy"] == 0.5 and m["defect_vs_background_accuracy"] == 1.0


# ----------------------------------------------------------------------------- tuning

def test_grid_configs_are_cartesian_product():
    cfg = ExperimentConfig.from_dict({"tune": {"rf_min_samples_leaf": [1, 3], "rf_max_features": ["sqrt", "log2"], "svm_C": [1.0, 5.0], "svm_gamma": ["scale"]}})
    rf = tr.grid_configs("rf", cfg)
    assert len(rf) == 4 and {(p["min_samples_leaf"], p["max_features"]) for p, _ in rf} == {(1, "sqrt"), (1, "log2"), (3, "sqrt"), (3, "log2")}
    for p, c in rf:
        assert c.random_forest.min_samples_leaf == p["min_samples_leaf"] and c.svm == cfg.svm
    svm = tr.grid_configs("svm", cfg)
    assert [p["C"] for p, _ in svm] == [1.0, 5.0] and svm[1][1].svm.C == 5.0 and svm[1][1].random_forest == cfg.random_forest
    with pytest.raises(ValueError, match="nepoznat model"):
        tr.grid_configs("knn", cfg)


@pytest.mark.parametrize("name", ["rf", "svm"])
def test_tune_model_picks_best_validation_score(name):
    cfg = ExperimentConfig.from_dict({**FAST, "tune": {"rf_min_samples_leaf": [1, 40], "rf_max_features": ["sqrt"], "svm_C": [0.001, 10.0], "svm_gamma": ["scale"]}})
    train, val = synthetic_ds(seed=0, group="g1", noise=3.0), synthetic_ds(seed=1, n_per_class=10, group="g2", noise=3.0)
    best, rows = tr.tune_model(name, cfg, train, val)
    assert len(rows) == 2 and rows[0]["macro_f1"] != rows[1]["macro_f1"]  # podaci su dovoljno teški da se kombinacije razlikuju
    top = max(rows, key=lambda r: r["macro_f1"])  # ugovor: bira se red s najboljim val macro F1
    chosen = {"min_samples_leaf": best.random_forest.min_samples_leaf, "max_features": best.random_forest.max_features} if name == "rf" \
        else {"C": best.svm.C, "gamma": best.svm.gamma}
    assert chosen == top["params"]
    if name == "rf":
        assert best.random_forest.min_samples_leaf == 1  # min_samples_leaf=40 ne može razdvojiti 7 klasa na 140 uzoraka
    assert best.split == cfg.split and best.candidates == cfg.candidates  # ostalo je netaknuto


# ----------------------------------------------------------------------------- run_training i CLI

def test_run_training_writes_models_and_summary(tmp_path):
    cfg = ExperimentConfig.from_dict({**FAST, "results_root": str(tmp_path)})
    train, val = synthetic_ds(seed=0, group="g1"), synthetic_ds(seed=1, n_per_class=10, group="g2")
    s = tr.run_training("rf", cfg, train, val, seeds=(0, 1))
    out = tmp_path / "exp1_rf"
    assert [r["seed"] for r in s["runs"]] == [0, 1] and s["n_train"] == 140 and s["n_val"] == 70 and s["feature_dim"] == 5
    assert s["train_class_counts"]["pozadina"] == 20 and s["val_macro_f1_mean"] > 0.9
    for seed in (0, 1):
        model, meta = load_model(out / f"model_seed{seed}")
        assert meta["seed"] == seed and meta["feature_names"] == train.feature_names
    assert json.loads((out / "train_summary.json").read_text())["model"] == "rf"
    assert ExperimentConfig.load(out / "config.json") == cfg


def test_run_training_refuses_overlapping_groups(tmp_path):
    cfg = ExperimentConfig.from_dict({**FAST, "results_root": str(tmp_path)})
    with pytest.raises(ValueError, match="curenje podataka"):
        tr.run_training("rf", cfg, synthetic_ds(group="g1"), synthetic_ds(group="g1"))
    assert not (tmp_path / "exp1_rf").exists()  # ništa nije upisano


def test_run_training_uses_configured_seeds(tmp_path):
    cfg = ExperimentConfig.from_dict({**FAST, "results_root": str(tmp_path), "train": {"rf_seeds": [3], "svm_seeds": [4]}})
    s = tr.run_training("rf", cfg, synthetic_ds(group="g1"), synthetic_ds(seed=1, group="g2"))
    assert [r["seed"] for r in s["runs"]] == [3] and (tmp_path / "exp1_rf" / "model_seed3").is_dir()


def _cli(root, tmp_path, *extra, model="rf", config=None):
    cfg_path = tmp_path / "cfg.json"
    ExperimentConfig.from_dict({**FAST, **(config or {})}).save(cfg_path)
    return tr.main(["--model", model, "--config", str(cfg_path), "--root", str(root), "--results-root", str(tmp_path / "res"), *extra])


def test_cli_end_to_end_and_cache_reuse(root, tmp_path, capsys):
    assert _cli(root, tmp_path, "--seeds", "0", "1") == 0
    out = capsys.readouterr().out
    assert "izgrađeno" in out and "iz cache-a" not in out
    res = tmp_path / "res" / "exp1_rf"
    assert (res / "model_seed0" / "model.joblib").is_file() and (res / "model_seed1" / "model.json").is_file()
    summary = json.loads((res / "train_summary.json").read_text())
    assert summary["n_train"] > 0 and summary["tuning"] is None and len(summary["runs"]) == 2
    assert list((tmp_path / "res" / "exp1_cache").glob("features_train_*.npz"))
    assert _cli(root, tmp_path, "--seeds", "0") == 0
    assert capsys.readouterr().out.count("iz cache-a") == 2  # train i val


def test_cli_tune_stores_chosen_hyperparameters(root, tmp_path):
    grid = {"tune": {"rf_min_samples_leaf": [1, 2], "rf_max_features": ["sqrt"]}}
    assert _cli(root, tmp_path, "--tune", "--seeds", "0", config=grid) == 0
    res = tmp_path / "res" / "exp1_rf"
    summary = json.loads((res / "train_summary.json").read_text())
    assert len(summary["tuning"]) == 2
    assert ExperimentConfig.load(res / "config.json").random_forest.min_samples_leaf in (1, 2)


def test_cli_trains_svm(root, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # SVC(probability=True) u novijem scikit-learn
        assert _cli(root, tmp_path, model="svm") == 0
    assert (tmp_path / "res" / "exp1_svm" / "model_seed0" / "model.joblib").is_file()


def test_cli_rejects_unknown_model_and_missing_dataset(tmp_path):
    with pytest.raises(SystemExit):
        tr.main(["--model", "knn"])
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        tr.main(["--model", "rf", "--root", str(tmp_path / "nema"), "--results-root", str(tmp_path / "res")])
