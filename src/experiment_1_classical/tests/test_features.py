"""Testovi ekstrakcije karakteristika (sintetičke slike, bez dataseta)."""
import numpy as np
import pytest

from src.experiment_1_classical.candidates import extract_candidates
from src.experiment_1_classical.config import FEATURE_GROUPS, FeatureConfig
from src.experiment_1_classical.features import POLARITY_NAMES, TOPOLOGY_NAMES, extract_features, feature_names, hog_length

from .synthetic import make_template


def feats(test, temp, cfg=None):
    """(X, imena kolona) za sve kandidate para; X se indeksira imenom pomoću col()."""
    cfg = cfg or FeatureConfig()
    cand = extract_candidates(test, temp)
    return extract_features(test, temp, cand, cfg), feature_names(cfg)


def col(X, names, name):
    return X[:, names.index(name)]


def bars_template():
    """Dva paralelna crna voda (redovi 100-110 i 130-140) na bijeloj podlozi."""
    img = np.full((640, 640), 255, dtype=np.uint8)
    img[100:110, 50:600] = 0
    img[130:140, 50:600] = 0
    return img


# ----------------------------------------------------------------------------- oblik i imena

@pytest.mark.parametrize("groups", [FEATURE_GROUPS, ("geometry",), ("polarity", "topology"), ("hog",), ("topology", "geometry")])
def test_shape_names_and_dtype(groups):
    temp = make_template()
    test = temp.copy()
    test[300:320, 300:320] = 0
    test[500:510, 100:130] = 0
    cfg = FeatureConfig(groups=groups)
    X, names = feats(test, temp, cfg)
    assert X.shape == (2, len(names)) and X.dtype == np.float32
    assert len(set(names)) == len(names) and np.isfinite(X).all()


def test_default_dimensions():
    assert hog_length(FeatureConfig()) == 324  # 3x3 bloka x 4 ćelije x 9 smjerova
    assert len(feature_names()) == 8 + 4 + 9 + 2 * 324


def test_hog_length_follows_config():
    cfg = FeatureConfig(crop_size=16, hog_pixels_per_cell=4, hog_orientations=6)
    assert hog_length(cfg) == 9 * 4 * 6
    temp = make_template()
    test = temp.copy()
    test[300:320, 300:320] = 0
    X, names = feats(test, temp, FeatureConfig(groups=("hog",), crop_size=16, hog_pixels_per_cell=4, hog_orientations=6))
    assert X.shape == (1, 2 * 216) and len(names) == 432


def test_groups_select_columns_in_order():
    temp = make_template()
    test = temp.copy()
    test[300:320, 300:320] = 0
    full, full_names = feats(test, temp)
    sub, sub_names = feats(test, temp, FeatureConfig(groups=("topology", "polarity")))
    assert sub_names == TOPOLOGY_NAMES + POLARITY_NAMES  # redoslijed prati cfg.groups
    for name in sub_names:
        assert np.allclose(col(sub, sub_names, name), col(full, full_names, name))


def test_no_candidates_gives_empty_matrix():
    temp = make_template()
    X, names = feats(temp, temp.copy())
    assert X.shape == (0, len(names))


# ----------------------------------------------------------------------------- geometrija i polaritet

def test_geometry_of_square_blob():
    temp = make_template()
    test = temp.copy()
    test[300:320, 300:320] = 0
    X, n = feats(test, temp)
    assert col(X, n, "geo_area")[0] == 400 and col(X, n, "geo_width")[0] == 20 and col(X, n, "geo_height")[0] == 20
    assert col(X, n, "geo_aspect")[0] == 1 and col(X, n, "geo_extent")[0] == 1 and col(X, n, "geo_solidity")[0] == 1


def test_geometry_of_wide_blob():
    temp = make_template()
    test = temp.copy()
    test[300:310, 300:340] = 0
    X, n = feats(test, temp)
    assert col(X, n, "geo_aspect")[0] == 4 and col(X, n, "geo_area")[0] == 400


def test_polarity_added_vs_removed_copper():
    temp = make_template()
    added = temp.copy()
    added[300:320, 300:320] = 0  # višak bakra na podlozi
    removed = temp.copy()
    removed[100:110, 300:306] = 255  # prekid u vodu
    Xa, n = feats(added, temp)
    Xr, _ = feats(removed, temp)
    assert col(Xa, n, "pol_added")[0] == 1 and col(Xa, n, "pol_removed")[0] == 0
    assert col(Xr, n, "pol_added")[0] == 0 and col(Xr, n, "pol_removed")[0] == 1
    assert col(Xa, n, "pol_mean_test")[0] == 0 and col(Xa, n, "pol_mean_temp")[0] == 1


# ----------------------------------------------------------------------------- topologija

def test_topology_short_merges_two_traces():
    temp = bars_template()
    test = temp.copy()
    test[110:130, 300:310] = 0  # most između vodova
    X, n = feats(test, temp)
    assert len(X) == 1 and col(X, n, "topo_black_cc_temp")[0] == 2 and col(X, n, "topo_black_cc_test")[0] == 1
    assert col(X, n, "topo_black_cc_delta")[0] == -1


def test_topology_open_breaks_a_trace():
    temp = bars_template()
    test = temp.copy()
    test[100:110, 300:306] = 255  # procjep u vodu
    X, n = feats(test, temp)
    assert col(X, n, "topo_black_cc_temp")[0] == 1 and col(X, n, "topo_black_cc_test")[0] >= 2
    assert col(X, n, "topo_black_cc_delta")[0] >= 1


def test_topology_pinhole_adds_white_component():
    temp = np.full((640, 640), 255, dtype=np.uint8)
    temp[250:350, 250:350] = 0  # velika crna površina
    test = temp.copy()
    test[295:300, 295:300] = 255  # bijela rupica unutar nje
    X, n = feats(test, temp)
    assert col(X, n, "topo_white_cc_temp")[0] == 0 and col(X, n, "topo_white_cc_test")[0] == 1
    assert col(X, n, "topo_white_cc_delta")[0] == 1 and col(X, n, "topo_black_cc_delta")[0] == 0
    assert col(X, n, "topo_black_frac_delta")[0] < 0  # rupica smanjuje udio crnog


# ----------------------------------------------------------------------------- svojstva

def test_features_depend_only_on_local_crop():
    """Karakteristike kandidata se ne mijenjaju kad se negdje daleko pojavi drugi defekt."""
    temp = make_template()
    one = temp.copy()
    one[300:320, 300:320] = 0
    two = one.copy()
    two[500:510, 100:130] = 0
    X1, _ = feats(one, temp)
    X2, n = feats(two, temp)
    assert len(X1) == 1 and len(X2) == 2
    assert np.array_equal(X1[0], X2[np.argmax(col(X2, n, "geo_area"))])  # veći (400 px) je isti kandidat


def test_hog_differs_for_different_content_and_is_deterministic():
    temp = make_template()
    sq, bar = temp.copy(), temp.copy()
    sq[300:320, 300:320] = 0
    bar[300:305, 300:340] = 0
    cfg = FeatureConfig(groups=("hog",))
    a1, _ = feats(sq, temp, cfg)
    a2, _ = feats(sq, temp, cfg)
    b, _ = feats(bar, temp, cfg)
    assert np.array_equal(a1, a2) and not np.allclose(a1, b)


def test_inputs_are_not_modified():
    temp = make_template()
    test = temp.copy()
    test[300:320, 300:320] = 0
    t0, p0 = test.copy(), temp.copy()
    feats(test, temp)
    assert np.array_equal(test, t0) and np.array_equal(temp, p0)


def test_invalid_inputs_raise():
    temp = make_template()
    test = temp.copy()
    test[300:320, 300:320] = 0
    cand = extract_candidates(test, temp)
    with pytest.raises(ValueError):
        extract_features(test.astype(float), temp, cand)  # pogrešan tip
    with pytest.raises(ValueError, match="oblika"):
        extract_features(test[:320, :320], temp[:320, :320], cand)  # kandidati iz druge veličine slike
