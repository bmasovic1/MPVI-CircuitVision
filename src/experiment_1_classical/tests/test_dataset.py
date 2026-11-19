"""Testovi učitavanja i provjere dataseta (lažna DeepPCB struktura, bez pravog dataseta)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.evaluation.metrics import get_split_items, image_id, load_ground_truth
from src.experiment_1_classical import dataset as ds
from src.experiment_1_classical.config import ExperimentConfig

from .synthetic import make_dataset

TRAINVAL = {"group00041": 3, "group13000": 3, "group20085": 3, "group44000": 3, "group50600": 3}
TEST = {"group00041": 2, "group12100": 2}


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("DeepPCB"), TRAINVAL, TEST)


@pytest.fixture(scope="module")
def cfg(root) -> ExperimentConfig:
    return ExperimentConfig(data_root=str(root))


# ----------------------------------------------------------------------------- load_gray

def test_load_gray_returns_uint8_2d(root):
    item = ds.get_items("val", ExperimentConfig(data_root=str(root)))[0]
    img = ds.load_gray(item["test"])
    assert img.shape == (640, 640) and img.dtype == np.uint8


def test_load_gray_converts_rgb_to_single_channel(tmp_path):
    p = tmp_path / "rgb.jpg"
    Image.fromarray(np.full((640, 640, 3), 200, dtype=np.uint8), mode="RGB").save(p)
    img = ds.load_gray(p)
    assert img.shape == (640, 640) and abs(int(img.mean()) - 200) <= 2


def test_load_gray_rejects_wrong_size(tmp_path):
    p = tmp_path / "small.png"
    Image.fromarray(np.zeros((100, 100), dtype=np.uint8)).save(p)
    with pytest.raises(ValueError, match="640x640"):
        ds.load_gray(p)


def test_load_gray_missing_and_corrupt(tmp_path):
    with pytest.raises(FileNotFoundError):
        ds.load_gray(tmp_path / "nema.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"ovo nije slika")
    with pytest.raises(ValueError, match="ne može pročitati"):
        ds.load_gray(bad)


# ----------------------------------------------------------------------------- load_annotations

def _ann(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "a.txt"
    p.write_text(text)
    return p


def test_load_annotations_valid(tmp_path):
    boxes, cls = ds.load_annotations(_ann(tmp_path, "10 20 50 60 3\n\n100 100 140 130 6\n"))
    assert boxes.shape == (2, 4) and boxes.dtype == float and cls.tolist() == [3, 6]
    assert boxes[0].tolist() == [10, 20, 50, 60]


def test_load_annotations_empty_file_gives_empty_arrays(tmp_path):
    boxes, cls = ds.load_annotations(_ann(tmp_path, ""))
    assert boxes.shape == (0, 4) and cls.shape == (0,)


@pytest.mark.parametrize("text", [
    "10 20 50 60 7",  # klasa izvan 1-6
    "10 20 50 60 0",  # klasa 0 se ne koristi
    "10 20 50 60",  # nedostaje tip
    "10,20,50,60,3",  # zarez umjesto razmaka (README DeepPCB-a pogrešno navodi zarez)
    "10 20 50 60 x",  # nije broj
    "50 20 10 60 3",  # x1 >= x2
    "10 60 50 20 3",  # y1 >= y2
    "10 20 50 641 3",  # izvan slike
    "-1 20 50 60 3",  # izvan slike
])
def test_load_annotations_rejects_invalid(tmp_path, text):
    with pytest.raises(ValueError, match="neispravna anotacija"):
        ds.load_annotations(_ann(tmp_path, text))


def test_load_annotations_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ds.load_annotations(tmp_path / "nema.txt")


# ----------------------------------------------------------------------------- get_items i podjela

def test_get_items_sizes_and_group_disjoint(cfg):
    train, val, test = (ds.get_items(s, cfg) for s in ("train", "val", "test"))
    assert len(train) + len(val) == sum(TRAINVAL.values()) and len(test) == sum(TEST.values())
    assert len(val) == 3  # jedna grupa od 3 slike (val_fraction 0.2 od 15)
    ds.ensure_group_disjoint(train, val)  # ne smije baciti grešku


def test_get_items_matches_shared_evaluation_split(cfg, root):
    """Isti val skup kao u src/evaluation (ground truth i image_id se poklapaju)."""
    ours = {image_id(i) for i in ds.get_items("val", cfg)}
    assert ours == set(load_ground_truth("val", root)) == {image_id(i) for i in get_split_items("val", root)}


def test_get_items_rejects_unknown_split(cfg):
    with pytest.raises(ValueError, match="nepoznat skup"):
        ds.get_items("trainval", cfg)


def test_get_items_missing_dataset_mentions_download_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        ds.get_items("val", ExperimentConfig(data_root=str(tmp_path / "nema")))


def test_get_items_single_group_leaves_train_empty(tmp_path):
    root = make_dataset(tmp_path, {"group00041": 4}, {"group00041": 1})
    with pytest.raises(ValueError, match="prazan"):
        ds.get_items("train", ExperimentConfig(data_root=str(root)))


def test_ensure_group_disjoint_detects_leakage(cfg):
    train = ds.get_items("train", cfg)
    with pytest.raises(ValueError, match="curenje podataka"):
        ds.ensure_group_disjoint(train, train[:1])


def test_test_split_may_share_groups_with_train(cfg):
    """Test dijeli grupe s trainval po dizajnu (docs/PLAN.md), pa se disjunktnost za test ne provjerava."""
    train_groups = {i["group"] for i in ds.get_items("train", cfg)}
    test_groups = {i["group"] for i in ds.get_items("test", cfg)}
    assert "group12100" in test_groups and test_groups & (train_groups | {i["group"] for i in ds.get_items("val", cfg)})


# ----------------------------------------------------------------------------- parovi, provjere, sažetak

def test_load_pair_and_iter_pairs(cfg):
    items = ds.get_items("val", cfg)
    gen = ds.iter_pairs(items)
    assert iter(gen) is gen  # generator, ne lista
    pairs = list(gen)
    assert len(pairs) == len(items)
    p = pairs[0]
    assert p.test.shape == p.temp.shape == (640, 640) and p.test.dtype == p.temp.dtype == np.uint8
    assert p.boxes.tolist() == [[300, 300, 330, 330]] and p.cls.tolist() == [5]
    assert p.group == items[0]["group"] and p.image_id == image_id(items[0])
    assert int(p.test[310, 310]) < 50 < int(p.temp[310, 310])  # defekt je samo na test slici


def test_check_items_reports_missing_files(tmp_path):
    root = make_dataset(tmp_path, {"group00041": 1, "group13000": 1}, {"group12100": 1})
    items = ds.get_items("test", ExperimentConfig(data_root=str(root)))
    assert ds.check_items(items) == []
    items[0]["temp"].unlink()
    problems = ds.check_items(items)
    assert len(problems) == 1 and "temp" in problems[0]


def test_summarize_items(cfg):
    s = ds.summarize_items(ds.get_items("test", cfg))
    assert s["n_pairs"] == 4 and s["n_defects"] == 4
    assert s["groups"] == {"group00041": 2, "group12100": 2}
    assert s["per_class"]["copper"] == 4 and sum(s["per_class"].values()) == 4


def test_cli_ok_and_missing(root, tmp_path, capsys):
    assert ds.main(["--split", "val", "--root", str(root), "--check-images"]) == 0
    assert "sve slike su ispravne" in capsys.readouterr().out
    broken = make_dataset(tmp_path, {"group00041": 1, "group13000": 1}, {"group12100": 1})
    next((broken / "PCBData" / "group12100").rglob("*_temp.jpg")).unlink()
    assert ds.main(["--split", "test", "--root", str(broken)]) == 1
    assert "PROBLEMI" in capsys.readouterr().out
