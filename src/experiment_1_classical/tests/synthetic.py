"""Pomoćne funkcije za testove: sintetičke slike i lažna DeepPCB struktura (bez pravog dataseta)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 640


def make_template() -> np.ndarray:
    """Binarna 'ploča': bijela podloga (255) s dva crna voda (0)."""
    img = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    img[100:110, 50:600] = 0
    img[200:210, 50:600] = 0
    return img


def make_dataset(root: Path, trainval: dict[str, int], test: dict[str, int]) -> Path:
    """Pravi minimalnu DeepPCB strukturu u `root` (root/PCBData/...) i vraća `root`.

    trainval, test: {naziv grupe, npr. 'group00041': broj parova}. Svaki par ima jedan defekt klase 3 na (300, 300)-(330, 330).
    Test slike dobijaju indekse od 500, da se ne sudare s trainval slikama iste grupe.
    """
    data = Path(root) / "PCBData"
    for split, spec, start in (("trainval", trainval, 0), ("test", test, 500)):
        lines = []
        for group, n in spec.items():
            prefix = group.removeprefix("group")
            (data / group / prefix).mkdir(parents=True, exist_ok=True)
            (data / group / f"{prefix}_not").mkdir(parents=True, exist_ok=True)
            for k in range(start, start + n):
                stem = f"{prefix}{k:03d}"
                temp = make_template()
                test_img = temp.copy()
                test_img[300:330, 300:330] = 0  # 'copper': višak bakra na podlozi
                Image.fromarray(temp).save(data / group / prefix / f"{stem}_temp.jpg", quality=95)
                Image.fromarray(test_img).save(data / group / prefix / f"{stem}_test.jpg", quality=95)
                (data / group / f"{prefix}_not" / f"{stem}.txt").write_text("300 300 330 330 5\n")
                lines.append(f"{group}/{prefix}/{stem}.jpg {group}/{prefix}_not/{stem}.txt")
        (data / f"{split}.txt").write_text("\n".join(lines) + "\n")
    return Path(root)
