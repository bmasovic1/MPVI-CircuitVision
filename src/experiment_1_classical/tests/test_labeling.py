"""Testovi označavanja kandidata i statistike lokalizacije (sintetički podaci, bez dataseta)."""
import numpy as np
import pytest

from src.experiment_1_classical import labeling as lb
from src.experiment_1_classical.config import BACKGROUND, ExperimentConfig
from src.experiment_1_classical.dataset import PcbPair

from .synthetic import make_dataset, make_template


def make_pair(blobs, gts, image_id="x", group="group00001") -> PcbPair:
    """blobs: [(r1, r2, c1, c2)] crne mrlje na test slici; gts: [(x1, y1, x2, y2, klasa)]."""
    temp = make_template()
    test = temp.copy()
    for r1, r2, c1, c2 in blobs:
        test[r1:r2, c1:c2] = 0
    boxes = np.array([g[:4] for g in gts], dtype=float).reshape(-1, 4)
    return PcbPair(image_id, group, test, temp, boxes, np.array([g[4] for g in gts], dtype=int))


# ----------------------------------------------------------------------------- label_candidates

def test_label_hits_misses_and_background():
    cand = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]], float)
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], float)
    labels = lb.label_candidates(cand, gt, np.array([3, 6]))
    assert labels.tolist() == [3, 6, BACKGROUND]


def test_label_threshold_boundary():
    cand = np.array([[0, 0, 10, 10]], float)
    gt = np.array([[0, 0, 10, 20]], float)  # IoU = 0.5 tačno
    assert lb.label_candidates(cand, gt, np.array([2]), iou_thr=0.5).tolist() == [2]
    assert lb.label_candidates(cand, gt, np.array([2]), iou_thr=0.51).tolist() == [BACKGROUND]


def test_label_picks_best_overlapping_gt():
    cand = np.array([[0, 0, 10, 10]], float)
    gt = np.array([[2, 0, 12, 10], [0, 0, 10, 10]], float)  # drugi ima IoU 1, prvi 2/3
    assert lb.label_candidates(cand, gt, np.array([1, 4])).tolist() == [4]


def test_label_many_candidates_can_share_one_gt():
    cand = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], float)
    gt = np.array([[0, 0, 10, 10]], float)
    assert lb.label_candidates(cand, gt, np.array([5])).tolist() == [5, 5]


def test_label_empty_inputs():
    assert lb.label_candidates(np.zeros((0, 4)), np.array([[0, 0, 5, 5]], float), np.array([1])).shape == (0,)
    assert lb.label_candidates(np.array([[0, 0, 5, 5]], float), np.zeros((0, 4)), np.zeros(0)).tolist() == [BACKGROUND]


@pytest.mark.parametrize("kwargs", [
    dict(gt_cls=np.array([1, 2])),  # broj klasa se ne poklapa s brojem GT boxova
    dict(gt_cls=np.array([1]), iou_thr=0.0),
    dict(gt_cls=np.array([1]), iou_thr=1.5),
])
def test_label_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        lb.label_candidates(np.array([[0, 0, 5, 5]], float), np.array([[0, 0, 5, 5]], float), **kwargs)


# ----------------------------------------------------------------------------- label_pair i statistika

def test_label_pair_marks_defect_and_spurious_blob():
    """Mrlja 20x20 na (300, 300) ima GT; box kandidata je mrlja + 10 px, što je tačno GT box (290..330)."""
    pair = make_pair([(300, 320, 300, 320), (500, 520, 100, 120)], [(290, 290, 330, 330, 5)])
    cand, labels = lb.label_pair(pair)
    assert len(cand) == 2
    by_box = {tuple(b): lab for b, lab in zip(cand.boxes.tolist(), labels.tolist())}
    assert by_box[(290.0, 290.0, 330.0, 330.0)] == 5 and by_box[(90.0, 490.0, 130.0, 530.0)] == BACKGROUND


def test_candidate_stats_counts_and_recall():
    hit = make_pair([(300, 320, 300, 320), (500, 520, 100, 120)], [(290, 290, 330, 330, 5)], "a")
    miss = make_pair([], [(100, 100, 140, 140, 3)], "b")  # defekt bez ikakve razlike -> nema kandidata
    st = lb.candidate_stats([hit, miss])
    assert st["n_images"] == 2 and st["n_gt"] == 2 and st["n_candidates"] == 2
    assert st["candidates_per_image"] == 1.0 and st["positive_fraction"] == 0.5
    assert st["label_counts"]["pozadina"] == 1 and st["label_counts"]["copper"] == 1
    r = st["recall"][0.5]
    assert r["all"] == 0.5 and r["copper"] == 1.0 and r["mousebite"] == 0.0 and np.isnan(r["open"])
    assert st["recall"][lb.PAPER_IOU]["all"] == 0.5


def test_candidate_stats_looser_threshold_recovers_loose_boxes():
    pair = make_pair([(300, 320, 300, 320)], [(280, 280, 340, 340, 2)])  # kandidat 40x40, GT 60x60: IoU = 1600/3600 = 0.44
    st = lb.candidate_stats([pair])
    assert st["recall"][0.5]["all"] == 0.0 and st["recall"][lb.PAPER_IOU]["all"] == 1.0
    assert st["positive_fraction"] == 0.0  # pri label_iou 0.5 kandidat je pozadina


def test_candidate_stats_empty_iterable():
    st = lb.candidate_stats([])
    assert st["n_images"] == 0 and np.isnan(st["recall"][0.5]["all"]) and np.isnan(st["candidates_per_image"])


def test_format_stats_contains_key_numbers():
    pair = make_pair([(300, 320, 300, 320)], [(290, 290, 330, 330, 5)])
    text = lb.format_stats(lb.candidate_stats([pair]), title="val: ")
    assert text.startswith("val: 1 slika, 1 defekata, 1 kandidata") and "100.0%" in text and "copper" in text


# ----------------------------------------------------------------------------- CLI

def test_cli_runs_on_fake_dataset(tmp_path, capsys):
    root = make_dataset(tmp_path, {"group00041": 2, "group13000": 2, "group20085": 2}, {"group12100": 1})
    assert lb.main(["--split", "train", "--root", str(root), "--step", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("train: ") and "Recall lokalizacije" in out


def test_cli_rejects_test_split_and_bad_step(tmp_path):
    with pytest.raises(SystemExit):
        lb.main(["--split", "test"])  # test se ne koristi za podešavanje parametara
    with pytest.raises(SystemExit):
        lb.main(["--step", "0"])


def test_config_label_iou_is_used():
    cfg = ExperimentConfig.from_dict({"candidates": {"label_iou": 0.9}})
    pair = make_pair([(300, 320, 300, 320)], [(286, 286, 330, 330, 5)])  # kandidat 40x40, GT 44x44: IoU = 1600/1936 = 0.83
    _, strict = lb.label_pair(pair, cfg)
    _, normal = lb.label_pair(pair)
    assert strict.tolist() == [BACKGROUND] and normal.tolist() == [5]
