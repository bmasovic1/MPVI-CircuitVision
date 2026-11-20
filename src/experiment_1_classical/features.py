"""Ekstrakcija karakteristika (features) po kandidatu za RF i SVM.

Sve karakteristike se računaju lokalno, iz isječka slike unutar (proširenog) boxa kandidata, pa ne zavise od
ostatka slike ni od položaja na ploči. Slike su binarne (crno = bakar, bijelo = podloga), pa se isječci
binarizuju pragom 128. Grupe (config.FeatureConfig.groups; izostavljanje grupe daje ablaciju):

  geometry  oblik komponente razlike: površina, širina, visina, omjer, popunjenost, obim, solidnost, kružnost
  polarity  smjer promjene: udio piksela gdje je test crn a template bijel (dodan bakar: copper, spur, short)
            naspram obrnutog (nedostaje bakar: open, mousebite, pin-hole), te srednje svjetline
  topology  broj povezanih crnih i bijelih komponenti u isječku na test i template slici i njihova razlika
            (short spaja vodove, open ih raskida, pin-hole dodaje bijelu komponentu), udio crnih piksela
  hog       HOG deskriptor isječka na test i template slici, svedenih na crop_size x crop_size (OpenCV HOGDescriptor)

Primjer:
    cand = extract_candidates(pair.test, pair.temp, cfg.candidates)
    X = extract_features(pair.test, pair.temp, cand, cfg.features)     # (N, D) float32
    names = feature_names(cfg.features)                                # D naziva, isti redoslijed kao kolone
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .candidates import Candidates, check_image_pair
from .config import FeatureConfig

BINARY_THRESHOLD = 128  # slike su binarne (DeepPCB); ispod praga je bakar (crno)

GEOMETRY_NAMES = ["geo_area", "geo_width", "geo_height", "geo_aspect", "geo_extent", "geo_perimeter",
                  "geo_solidity", "geo_circularity"]
POLARITY_NAMES = ["pol_added", "pol_removed", "pol_mean_test", "pol_mean_temp"]
TOPOLOGY_NAMES = ["topo_black_cc_test", "topo_black_cc_temp", "topo_black_cc_delta",
                  "topo_white_cc_test", "topo_white_cc_temp", "topo_white_cc_delta",
                  "topo_black_frac_test", "topo_black_frac_temp", "topo_black_frac_delta"]


def hog_length(cfg: FeatureConfig) -> int:
    """Dužina HOG vektora jedne slike: blokovi 2x2 ćelije s korakom od jedne ćelije."""
    cells = cfg.crop_size // cfg.hog_pixels_per_cell
    return (cells - 1) ** 2 * 4 * cfg.hog_orientations


def feature_names(cfg: FeatureConfig | None = None) -> list[str]:
    """Nazivi kolona matrice karakteristika, u istom redoslijedu kao u extract_features."""
    cfg = cfg or FeatureConfig()
    names: list[str] = []
    for group in cfg.groups:
        if group == "geometry":
            names += GEOMETRY_NAMES
        elif group == "polarity":
            names += POLARITY_NAMES
        elif group == "topology":
            names += TOPOLOGY_NAMES
        else:
            names += [f"hog_{src}_{i}" for src in ("test", "temp") for i in range(hog_length(cfg))]
    return names


# ----------------------------------------------------------------------------- grupe karakteristika

def _geometry(mask: np.ndarray) -> list[float]:
    """Oblik komponente (bool maska isječka)."""
    ys, xs = np.nonzero(mask)
    area = float(len(xs))
    w, h = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea) if len(contours) > 1 else contours[0]
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = min(area / hull_area, 1.0) if hull_area > 0 else 1.0
    circularity = 4.0 * math.pi * area / perimeter ** 2 if perimeter > 0 else 0.0
    return [area, w, h, w / h, area / (w * h), perimeter, solidity, circularity]


def _polarity(test: np.ndarray, temp: np.ndarray, mask: np.ndarray) -> list[float]:
    """Smjer promjene unutar komponente: dodan bakar (test crn, template bijel) ili nedostaje bakar."""
    t, p = test[mask], temp[mask]
    t_black, p_black = t < BINARY_THRESHOLD, p < BINARY_THRESHOLD
    return [float(np.mean(t_black & ~p_black)), float(np.mean(~t_black & p_black)),
            float(t.mean()) / 255.0, float(p.mean()) / 255.0]


def _count_components(binary: np.ndarray, connectivity: int) -> int:
    return int(cv2.connectedComponents(binary.astype(np.uint8), connectivity=connectivity)[0]) - 1


def _topology(test: np.ndarray, temp: np.ndarray) -> list[float]:
    """Broj povezanih komponenti u isječku: crne (bakar, 8-susjedstvo) i bijele (podloga, 4-susjedstvo)."""
    t_black, p_black = test < BINARY_THRESHOLD, temp < BINARY_THRESHOLD
    out: list[float] = []
    for binary_t, binary_p, conn in ((t_black, p_black, 8), (~t_black, ~p_black, 4)):
        n_t, n_p = _count_components(binary_t, conn), _count_components(binary_p, conn)
        out += [float(n_t), float(n_p), float(n_t - n_p)]
    f_t, f_p = float(t_black.mean()), float(p_black.mean())
    return out + [f_t, f_p, f_t - f_p]


def _hog(hog: cv2.HOGDescriptor, crop: np.ndarray, size: int) -> np.ndarray:
    return hog.compute(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)).ravel()


# ----------------------------------------------------------------------------- glavna funkcija

def extract_features(test: np.ndarray, temp: np.ndarray, cand: Candidates, cfg: FeatureConfig | None = None) -> np.ndarray:
    """Matrica karakteristika (N, D) float32, jedan red po kandidatu, kolone prema feature_names(cfg)."""
    cfg = cfg or FeatureConfig()
    check_image_pair(test, temp)
    if cand.labels.shape != test.shape:
        raise ValueError(f"kandidati su izdvojeni iz slike oblika {cand.labels.shape}, a slika je {test.shape}")
    p = cfg.hog_pixels_per_cell
    hog = cv2.HOGDescriptor((cfg.crop_size,) * 2, (2 * p,) * 2, (p, p), (p, p), cfg.hog_orientations) if "hog" in cfg.groups else None

    rows = np.zeros((len(cand), len(feature_names(cfg))), dtype=np.float32)
    for k in range(len(cand)):
        x1, y1, x2, y2 = (int(v) for v in cand.boxes[k])
        t_crop, p_crop = test[y1:y2, x1:x2], temp[y1:y2, x1:x2]
        mask = cand.labels[y1:y2, x1:x2] == cand.component_ids[k]
        values: list[float] = []
        for group in cfg.groups:
            if group == "geometry":
                values += _geometry(mask)
            elif group == "polarity":
                values += _polarity(t_crop, p_crop, mask)
            elif group == "topology":
                values += _topology(t_crop, p_crop)
            else:
                values += _hog(hog, t_crop, cfg.crop_size).tolist() + _hog(hog, p_crop, cfg.crop_size).tolist()
        rows[k] = values
    return rows
