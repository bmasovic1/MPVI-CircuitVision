"""Konfiguracija Eksperimenta 1 (OpenCV + Random Forest/SVM): svi hiperparametri na jednom mjestu.

Konfiguracija je skup nepromjenjivih (frozen) dataclass-a koji se validiraju pri kreiranju, mogu se
upisati u JSON (uz model, radi reproducibilnosti) i učitati nazad:

    cfg = ExperimentConfig()                       # početne vrijednosti
    cfg = ExperimentConfig.from_dict({"candidates": {"diff_threshold": 80}})   # djelimične izmjene
    cfg.save("results/exp1_rf/config.json"); ExperimentConfig.load("results/exp1_rf/config.json")

Napomena: vrijednosti za kandidate i karakteristike su POČETNE; kalibrišu se u koracima 3-7 na
validacijskom skupu (nikad na testu), a nova polja se dodaju tek kad ih neki korak počne koristiti.
Podjela train/val/test se NE definiše ovdje, nego u src/data/deeppcb.py (split_by_group); ovdje su
samo parametri koji moraju odgovarati toj podjeli.
"""
from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from src.data.deeppcb import DEFAULT_ROOT, NAMES

EXPERIMENT = "exp1"
VARIANTS = ("rf", "svm", "localization_only")  # varijante u results/results.csv (docs/FORMAT.md)
SPLITS = ("val", "test")  # skupovi za koje se pišu predikcije
FEATURE_GROUPS = ("geometry", "polarity", "topology", "hog")  # grupe karakteristika (features.py)
BACKGROUND = 0  # oznaka kandidata koji nije defekt (u predikcije se ne upisuje)
DEFECT_CLASSES = tuple(sorted(NAMES))  # (1, ..., 6) kao u DeepPCB-u

T = TypeVar("T")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_gamma(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return value in ("scale", "auto") or (isinstance(value, (int, float)) and value > 0)


def _require_odd(value: int, name: str) -> None:
    _require(isinstance(value, int) and value >= 1 and value % 2 == 1, f"{name} mora biti neparan cijeli broj >= 1 (dobijeno {value!r})")


# ----------------------------------------------------------------------------- dijelovi konfiguracije

@dataclass(frozen=True)
class SplitConfig:
    """Parametri podjele train/val; moraju biti isti kao u src.data.deeppcb.split_by_group."""
    val_fraction: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        _require(0.0 < self.val_fraction < 1.0, f"val_fraction mora biti u (0, 1), dobijeno {self.val_fraction!r}")


@dataclass(frozen=True)
class CandidateConfig:
    """Oduzimanje test i template slike i izdvajanje kandidata (korak 3)."""
    # Vrijednosti su odabrane na uzorku TRAIN skupa (svaka 5. slika; recall@IoU0.5 granice kandidata ~95%):
    diff_threshold: int = 100  # prag na |test - temp| (0-255); slike su binarne pa su pragovi 60-160 ekvivalentni
    open_kernel: int = 3  # morfološko otvaranje: uklanja tanke (1-2 px) konture oko vodova
    close_kernel: int = 5  # morfološko zatvaranje: spaja dijelove istog defekta
    min_area: int = 4  # najmanja površina komponente (px)
    max_area: int = 6000  # najveća površina komponente (px); veće su razlike strukture (npr. neporavnanje)
    box_padding: int = 10  # GT box je u prosjeku ~20 px veći od same razlike, tj. ~10 px sa svake strane
    label_iou: float = 0.5  # prag IoU s GT boxom za dodjelu klase kandidatu (isti kao mAP@0.5 u evaluaciji)

    def __post_init__(self) -> None:
        _require(0 <= self.diff_threshold <= 255, f"diff_threshold mora biti u [0, 255], dobijeno {self.diff_threshold!r}")
        _require_odd(self.open_kernel, "open_kernel")
        _require_odd(self.close_kernel, "close_kernel")
        _require(1 <= self.min_area < self.max_area, f"mora vrijediti 1 <= min_area < max_area (dobijeno {self.min_area!r}, {self.max_area!r})")
        _require(self.box_padding >= 0, f"box_padding mora biti >= 0, dobijeno {self.box_padding!r}")
        _require(0.0 < self.label_iou <= 1.0, f"label_iou mora biti u (0, 1], dobijeno {self.label_iou!r}")


@dataclass(frozen=True)
class FeatureConfig:
    """Ekstrakcija karakteristika po kandidatu (korak 5)."""
    groups: tuple[str, ...] = FEATURE_GROUPS  # koje grupe karakteristika se koriste (ablacija: izostaviti grupu)
    crop_size: int = 32  # stranica isječka (px) na koji se svodi box kandidata za HOG
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 8

    def __post_init__(self) -> None:
        _require(len(self.groups) >= 1 and len(set(self.groups)) == len(self.groups) and set(self.groups) <= set(FEATURE_GROUPS),
                 f"groups mora biti neprazan podskup od {FEATURE_GROUPS} bez ponavljanja, dobijeno {self.groups!r}")
        _require(self.crop_size >= 8, f"crop_size mora biti >= 8, dobijeno {self.crop_size!r}")
        _require(self.hog_orientations >= 1, f"hog_orientations mora biti >= 1, dobijeno {self.hog_orientations!r}")
        _require(1 <= self.hog_pixels_per_cell and self.crop_size % self.hog_pixels_per_cell == 0
                 and self.crop_size >= 2 * self.hog_pixels_per_cell,
                 "hog_pixels_per_cell mora dijeliti crop_size, a crop_size mora biti >= 2 * hog_pixels_per_cell (HOG blok je 2x2 ćelije)")


@dataclass(frozen=True)
class RandomForestConfig:
    """Hiperparametri Random Foresta (korak 6); seed se zadaje pri treniranju (TrainConfig.rf_seeds)."""
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    class_weight: str | None = "balanced"  # klasa "pozadina" je znatno brojnija od defekata
    n_jobs: int = -1

    def __post_init__(self) -> None:
        _require(self.n_estimators >= 1, f"n_estimators mora biti >= 1, dobijeno {self.n_estimators!r}")
        _require(self.max_depth is None or self.max_depth >= 1, f"max_depth mora biti None ili >= 1, dobijeno {self.max_depth!r}")
        _require(self.min_samples_leaf >= 1, f"min_samples_leaf mora biti >= 1, dobijeno {self.min_samples_leaf!r}")
        _require(self.max_features in ("sqrt", "log2"), f"max_features mora biti 'sqrt' ili 'log2', dobijeno {self.max_features!r}")
        _require(self.class_weight in (None, "balanced"), f"class_weight mora biti None ili 'balanced', dobijeno {self.class_weight!r}")


@dataclass(frozen=True)
class SvmConfig:
    """Hiperparametri SVM-a (korak 6); karakteristike se standardizuju u pipelineu (scaler se fituje samo na train)."""
    kernel: str = "rbf"
    C: float = 10.0
    gamma: str | float = "scale"  # 'scale', 'auto' ili pozitivan broj
    class_weight: str | None = "balanced"
    probability: bool = True  # potrebno za score u [0, 1] (docs/FORMAT.md)

    def __post_init__(self) -> None:
        _require(self.kernel in ("rbf", "linear"), f"kernel mora biti 'rbf' ili 'linear', dobijeno {self.kernel!r}")
        _require(self.C > 0, f"C mora biti > 0, dobijeno {self.C!r}")
        _require(_valid_gamma(self.gamma), f"gamma mora biti 'scale', 'auto' ili pozitivan broj, dobijeno {self.gamma!r}")
        _require(self.class_weight in (None, "balanced"), f"class_weight mora biti None ili 'balanced', dobijeno {self.class_weight!r}")


@dataclass(frozen=True)
class TrainConfig:
    """Seedovi treniranja: RF s 3 seeda (prosjek i odstupanje), SVM je deterministički pa jedan run (docs/FORMAT.md)."""
    rf_seeds: tuple[int, ...] = (0, 1, 2)
    svm_seeds: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        for name in ("rf_seeds", "svm_seeds"):
            seeds = getattr(self, name)
            _require(len(seeds) >= 1 and len(set(seeds)) == len(seeds) and all(isinstance(s, int) for s in seeds),
                     f"{name} mora biti neprazna lista različitih cijelih brojeva, dobijeno {seeds!r}")


@dataclass(frozen=True)
class TuneConfig:
    """Mreža hiperparametara za izbor na VALIDACIJSKOM skupu (train.py --tune); kriterij je macro F1 kandidata.

    Za gamma: 'scale' je 1/broj_karakteristika (nakon standardizacije), npr. ~0.0015 za 669 karakteristika.
    """
    rf_min_samples_leaf: tuple[int, ...] = (1, 2, 5)
    rf_max_features: tuple[str, ...] = ("sqrt", "log2")
    svm_C: tuple[float, ...] = (1.0, 10.0, 100.0)
    svm_gamma: tuple[str | float, ...] = ("scale", 0.0005, 0.005)

    def __post_init__(self) -> None:
        for name in ("rf_min_samples_leaf", "rf_max_features", "svm_C", "svm_gamma"):
            values = getattr(self, name)
            _require(len(values) >= 1 and len(set(values)) == len(values), f"{name} mora biti neprazan, bez ponavljanja, dobijeno {values!r}")
        _require(all(isinstance(v, int) and v >= 1 for v in self.rf_min_samples_leaf), "rf_min_samples_leaf: cijeli brojevi >= 1")
        _require(all(v in ("sqrt", "log2") for v in self.rf_max_features), "rf_max_features: 'sqrt' ili 'log2'")
        _require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in self.svm_C), "svm_C: pozitivni brojevi")
        _require(all(_valid_gamma(v) for v in self.svm_gamma), "svm_gamma: 'scale', 'auto' ili pozitivni brojevi")


# ----------------------------------------------------------------------------- cijela konfiguracija

@dataclass(frozen=True)
class ExperimentConfig:
    """Sve postavke Eksperimenta 1. Putanje su relativne i konfigurabilne (lokalno, Colab, Drive)."""
    data_root: str = DEFAULT_ROOT.as_posix()  # preuzima scripts/download_data.py
    results_root: str = "results"
    split: SplitConfig = field(default_factory=SplitConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    svm: SvmConfig = field(default_factory=SvmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)

    def results_dir(self, variant: str) -> Path:
        """Folder rezultata varijante, npr. results/exp1_rf (docs/FORMAT.md: results/<eksperiment>/)."""
        _require(variant in VARIANTS, f"nepoznata varijanta {variant!r}; dozvoljene: {VARIANTS}")
        return Path(self.results_root) / f"{EXPERIMENT}_{variant}"

    def predictions_path(self, variant: str, split: str) -> Path:
        """Putanja predictions_<skup>.json za varijantu."""
        _require(split in SPLITS, f"nepoznat skup {split!r}; dozvoljeni: {SPLITS}")
        return self.results_dir(variant) / f"predictions_{split}.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Gradi konfiguraciju iz rječnika; izostavljena polja dobijaju početne vrijednosti, nepoznata ključa su greška."""
        return _from_dict(cls, data)

    def save(self, path: str | Path) -> None:
        """Upisuje konfiguraciju u JSON (čuva se uz model i rezultate radi reproducibilnosti)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    _require(isinstance(data, dict), f"{cls.__name__}: očekivan rječnik, dobijeno {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - set(known))
    _require(not unknown, f"{cls.__name__}: nepoznata polja {unknown}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        factory = known[name].default_factory
        if factory is not MISSING and is_dataclass(factory):
            kwargs[name] = _from_dict(factory, value)  # ugniježđena sekcija
        elif isinstance(value, list):
            kwargs[name] = tuple(value)  # JSON nema tuple
        else:
            kwargs[name] = value
    return cls(**kwargs)
