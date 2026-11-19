"""Oduzimanje test i template slike i izdvajanje kandidata za defekte (OpenCV).

Postupak (parametri u config.CandidateConfig):
  1. |test - temp| (cv2.absdiff) i prag -> binarna maska razlike (razlika u oba smjera: višak i manjak bakra)
  2. morfološko otvaranje: uklanja tanke konture oko vodova (85% razlika nisu defekti, docs/DATASET.md)
  3. morfološko zatvaranje: spaja dijelove istog defekta
  4. povezane komponente (8-susjedstvo) i filtriranje po površini
  5. box komponente proširen za box_padding (GT boxovi su veći od same razlike) i ograničen na sliku

Poznato ograničenje: kod neporavnanih parova (test/template pomaknuti za nekoliko piksela) razlika stvara velike
komponente duž vodova koje se spoje s defektom i kandidat se promaši. Osnovna varijanta to ne rješava;
poravnanje je predviđeno kao posebna varijanta (`*_aligned`).

Primjer:
    cand = extract_candidates(pair.test, pair.temp, cfg.candidates)
    cand.boxes      # (N, 4) float: x1, y1, x2, y2 u pikselima, proširene i ograničene na sliku
    cand.mask(0)    # bool (640, 640): pikseli prve komponente
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CandidateConfig


@dataclass(frozen=True)
class Candidates:
    """Kandidati za defekte jednog para slika (redoslijed je determinističan: po oznaci komponente)."""
    boxes: np.ndarray  # (N, 4) float, x1, y1, x2, y2 (proširen box komponente)
    areas: np.ndarray  # (N,) int, površina komponente u pikselima (bez proširenja)
    component_ids: np.ndarray  # (N,) int, oznaka komponente u `labels`
    labels: np.ndarray  # (H, W) int32, slika povezanih komponenti (0 = pozadina)

    def __len__(self) -> int:
        return len(self.boxes)

    def mask(self, index: int) -> np.ndarray:
        """Bool maska piksela komponente kandidata `index` (za karakteristike u koraku 5)."""
        return self.labels == self.component_ids[index]


def _check_pair(test: np.ndarray, temp: np.ndarray) -> None:
    for name, img in (("test", test), ("temp", temp)):
        if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
            raise ValueError(f"{name} mora biti 2D numpy niz tipa uint8 (sivi tonovi)")
    if test.shape != temp.shape:
        raise ValueError(f"test i temp moraju imati isti oblik, dobijeno {test.shape} i {temp.shape}")


def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def difference_mask(test: np.ndarray, temp: np.ndarray, cfg: CandidateConfig | None = None) -> np.ndarray:
    """Binarna maska razlike (uint8, 0/1) nakon praga, otvaranja i zatvaranja."""
    cfg = cfg or CandidateConfig()
    _check_pair(test, temp)
    mask = (cv2.absdiff(test, temp) > cfg.diff_threshold).astype(np.uint8)
    if cfg.open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(cfg.open_kernel))
    if cfg.close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(cfg.close_kernel))
    return mask


def extract_candidates(test: np.ndarray, temp: np.ndarray, cfg: CandidateConfig | None = None) -> Candidates:
    """Izdvaja kandidate za defekte iz para slika (uint8, sivi tonovi, isti oblik)."""
    cfg = cfg or CandidateConfig()
    mask = difference_mask(test, temp, cfg)
    height, width = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]  # komponenta 0 je pozadina
    ids = np.flatnonzero((areas >= cfg.min_area) & (areas <= cfg.max_area)) + 1
    x, y = stats[ids, cv2.CC_STAT_LEFT], stats[ids, cv2.CC_STAT_TOP]
    w, h = stats[ids, cv2.CC_STAT_WIDTH], stats[ids, cv2.CC_STAT_HEIGHT]
    pad = cfg.box_padding
    boxes = np.stack([np.maximum(x - pad, 0), np.maximum(y - pad, 0),
                      np.minimum(x + w + pad, width), np.minimum(y + h + pad, height)], axis=1).astype(float)
    return Candidates(boxes=boxes.reshape(-1, 4), areas=stats[ids, cv2.CC_STAT_AREA].astype(int),
                      component_ids=ids.astype(int), labels=labels)
