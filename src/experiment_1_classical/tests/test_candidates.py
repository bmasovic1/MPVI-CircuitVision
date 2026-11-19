"""Testovi oduzimanja slika i izdvajanja kandidata (sintetičke slike, bez dataseta)."""
import numpy as np
import pytest

from src.experiment_1_classical.candidates import difference_mask, extract_candidates
from src.experiment_1_classical.config import CandidateConfig

from .synthetic import make_template


def pair(*edits):
    """Template i test slika; svaka izmjena je (red od, red do, kolona od, kolona do, vrijednost) na test slici."""
    temp = make_template()
    test = temp.copy()
    for r1, r2, c1, c2, v in edits:
        test[r1:r2, c1:c2] = v
    return test, temp


def test_identical_images_give_no_candidates():
    temp = make_template()
    cand = extract_candidates(temp, temp.copy())
    assert len(cand) == 0 and cand.boxes.shape == (0, 4) and cand.areas.shape == (0,)
    assert cand.labels.shape == (640, 640)


def test_single_added_blob_box_is_padded_bbox():
    test, temp = pair((300, 320, 300, 320, 0))  # crni 20x20 'višak bakra' na bijeloj podlozi
    cand = extract_candidates(test, temp)  # početne vrijednosti: padding 10
    assert len(cand) == 1
    assert cand.boxes[0].tolist() == [290, 290, 330, 330] and cand.areas[0] == 400
    assert cand.boxes.dtype == float


def test_missing_copper_is_detected_too():
    """Razlika je u oba smjera: bijela 'rupa' u crnom vodu (prekid) daje kandidat isto kao dodani bakar."""
    test, temp = pair((100, 110, 300, 306, 255))
    cand = extract_candidates(test, temp)
    assert len(cand) == 1 and cand.boxes[0].tolist() == [290, 90, 316, 120] and cand.areas[0] == 60


def test_box_is_clipped_to_image():
    test, temp = pair((0, 20, 0, 20, 0), (620, 640, 620, 640, 0))
    boxes = extract_candidates(test, temp).boxes
    assert sorted(map(tuple, boxes.tolist())) == [(0, 0, 30, 30), (610, 610, 640, 640)]


def test_thin_contours_removed_by_opening():
    """1 px linija (kao konture oko vodova u razlici) nestaje otvaranjem 3x3, a bez otvaranja ostaje."""
    test, temp = pair((300, 400, 300, 301, 0))
    assert len(extract_candidates(test, temp)) == 0
    assert len(extract_candidates(test, temp, CandidateConfig(open_kernel=1))) == 1


def test_closing_merges_nearby_fragments():
    test, temp = pair((300, 310, 300, 310, 0), (300, 310, 313, 323, 0))  # razmak od 3 px
    assert len(extract_candidates(test, temp)) == 1
    assert len(extract_candidates(test, temp, CandidateConfig(close_kernel=1))) == 2


def test_area_filters():
    test, temp = pair((300, 305, 100, 105, 0), (300, 310, 400, 410, 0), (500, 540, 300, 340, 0))  # 25, 100, 1600 px
    cfg = CandidateConfig(open_kernel=1, close_kernel=1, min_area=50, max_area=1000)
    assert extract_candidates(test, temp, cfg).areas.tolist() == [100]


def test_threshold_ignores_weak_differences():
    test, temp = pair((300, 320, 300, 320, 215))  # razlika 40 od 255
    assert len(extract_candidates(test, temp)) == 0  # prag 100
    assert len(extract_candidates(test, temp, CandidateConfig(diff_threshold=20))) == 1


def test_mask_and_labels_are_consistent():
    test, temp = pair((300, 320, 300, 320, 0), (500, 510, 100, 130, 0))
    cand = extract_candidates(test, temp)
    assert len(cand) == 2
    for k in range(len(cand)):
        m = cand.mask(k)
        assert m.dtype == bool and m.shape == (640, 640) and int(m.sum()) == cand.areas[k]
        ys, xs = np.nonzero(m)  # komponenta leži unutar svog (proširenog) boxa
        assert cand.boxes[k][0] <= xs.min() and xs.max() < cand.boxes[k][2] and cand.boxes[k][1] <= ys.min() and ys.max() < cand.boxes[k][3]


def test_difference_mask_is_binary_uint8():
    test, temp = pair((300, 320, 300, 320, 0))
    m = difference_mask(test, temp)
    assert m.dtype == np.uint8 and set(np.unique(m).tolist()) <= {0, 1} and int(m.sum()) == 400


def test_extraction_is_deterministic():
    test, temp = pair((300, 320, 300, 320, 0), (500, 510, 100, 130, 0))
    a, b = extract_candidates(test, temp), extract_candidates(test, temp)
    assert np.array_equal(a.boxes, b.boxes) and np.array_equal(a.labels, b.labels)


def test_inputs_are_not_modified():
    test, temp = pair((300, 320, 300, 320, 0))
    t0, p0 = test.copy(), temp.copy()
    extract_candidates(test, temp)
    assert np.array_equal(test, t0) and np.array_equal(temp, p0)


@pytest.mark.parametrize("bad_test, bad_temp", [
    (np.zeros((640, 640), dtype=float), np.zeros((640, 640), dtype=np.uint8)),  # pogrešan tip
    (np.zeros((640, 640, 3), dtype=np.uint8), np.zeros((640, 640), dtype=np.uint8)),  # nije 2D
    (np.zeros((640, 640), dtype=np.uint8), np.zeros((320, 320), dtype=np.uint8)),  # različit oblik
    ([[0]], np.zeros((640, 640), dtype=np.uint8)),  # nije numpy niz
])
def test_invalid_inputs_raise(bad_test, bad_temp):
    with pytest.raises(ValueError):
        extract_candidates(bad_test, bad_temp)
