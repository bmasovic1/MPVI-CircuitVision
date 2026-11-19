"""Testovi konfiguracije Eksperimenta 1 (ne trebaju dataset)."""
import dataclasses
import inspect
from pathlib import Path

import pytest

from src.data.deeppcb import NAMES, split_by_group
from src.experiment_1_classical import config as cfg_mod
from src.experiment_1_classical.config import (BACKGROUND, DEFECT_CLASSES, CandidateConfig, ExperimentConfig,
                                               FeatureConfig, RandomForestConfig, SplitConfig, SvmConfig, TrainConfig)


def test_default_config_is_valid():
    cfg = ExperimentConfig()
    assert cfg.split.val_fraction == 0.2 and cfg.train.rf_seeds == (0, 1, 2) and cfg.train.svm_seeds == (0,)


def test_split_defaults_match_shared_data_split():
    """Podjela mora biti ista kao u src/data (isti val skup za sve eksperimente)."""
    params = inspect.signature(split_by_group).parameters
    assert SplitConfig().val_fraction == params["val_fraction"].default
    assert SplitConfig().seed == params["seed"].default


def test_class_constants_follow_deeppcb():
    assert DEFECT_CLASSES == (1, 2, 3, 4, 5, 6) == tuple(sorted(NAMES))
    assert BACKGROUND not in DEFECT_CLASSES


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ExperimentConfig().candidates.diff_threshold = 5  # type: ignore[misc]


def test_json_roundtrip_preserves_config(tmp_path: Path):
    cfg = ExperimentConfig.from_dict({"candidates": {"diff_threshold": 80}, "train": {"rf_seeds": [5, 6]}})
    path = tmp_path / "sub" / "config.json"  # save mora napraviti folder
    cfg.save(path)
    loaded = ExperimentConfig.load(path)
    assert loaded == cfg
    assert loaded.train.rf_seeds == (5, 6)  # lista iz JSON-a vraćena u tuple


def test_from_dict_partial_keeps_defaults():
    cfg = ExperimentConfig.from_dict({"svm": {"C": 3.0}})
    assert cfg.svm.C == 3.0 and cfg.svm.kernel == SvmConfig().kernel
    assert cfg.candidates == CandidateConfig()


@pytest.mark.parametrize("data", [
    {"candidate": {}},  # tipfeler u nazivu sekcije
    {"candidates": {"diff_treshold": 5}},  # tipfeler u nazivu polja
    {"candidates": 5},  # sekcija nije rječnik
])
def test_from_dict_rejects_unknown_or_malformed(data):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(data)


@pytest.mark.parametrize("factory", [
    lambda: SplitConfig(val_fraction=0.0),
    lambda: SplitConfig(val_fraction=1.0),
    lambda: CandidateConfig(diff_threshold=256),
    lambda: CandidateConfig(diff_threshold=-1),
    lambda: CandidateConfig(open_kernel=4),  # paran kernel
    lambda: CandidateConfig(close_kernel=0),
    lambda: CandidateConfig(min_area=50, max_area=50),
    lambda: CandidateConfig(box_padding=-1),
    lambda: CandidateConfig(label_iou=0.0),
    lambda: CandidateConfig(label_iou=1.5),
    lambda: FeatureConfig(crop_size=4),
    lambda: FeatureConfig(crop_size=32, hog_pixels_per_cell=5),  # ne dijeli crop_size
    lambda: FeatureConfig(hog_orientations=0),
    lambda: RandomForestConfig(n_estimators=0),
    lambda: RandomForestConfig(max_depth=0),
    lambda: RandomForestConfig(class_weight="auto"),
    lambda: SvmConfig(C=0),
    lambda: SvmConfig(kernel="poly"),
    lambda: SvmConfig(gamma="x"),
    lambda: TrainConfig(rf_seeds=()),
    lambda: TrainConfig(rf_seeds=(1, 1)),
    lambda: TrainConfig(svm_seeds=(0.5,)),
])
def test_invalid_values_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_results_paths_follow_format_doc():
    cfg = ExperimentConfig(results_root="/drive/results")
    assert cfg.results_dir("rf") == Path("/drive/results/exp1_rf")
    assert cfg.predictions_path("svm", "val") == Path("/drive/results/exp1_svm/predictions_val.json")
    assert cfg.predictions_path("localization_only", "test").name == "predictions_test.json"


def test_results_paths_reject_unknown_names():
    cfg = ExperimentConfig()
    with pytest.raises(ValueError):
        cfg.results_dir("knn")
    with pytest.raises(ValueError):
        cfg.predictions_path("rf", "train")  # predikcije se pišu samo za val i test


def test_variants_listed_in_module():
    assert set(cfg_mod.VARIANTS) == {"rf", "svm", "localization_only"}
