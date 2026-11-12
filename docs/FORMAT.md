# Zajednički format predikcija i evaluacije

Svi eksperimenti (1-4) moraju pisati predikcije u istom formatu da bi `src/evaluation/` mogao poredati modele.

## Predikcije: `results/<eksperiment>/predictions_<skup>.json`
`<skup>` je `val` ili `test`. Jedan fajl, lista zapisa, jedan zapis po slici:

```json
[
  {
    "image": "20085000",
    "boxes": [
      {"x1": 409, "y1": 394, "x2": 435, "y2": 422, "score": 0.91, "cls": 3}
    ]
  }
]
```

- `image`: ime slike bez nastavka i bez `_test` (npr. `20085000`). Slika bez detekcija ima `"boxes": []` (mora biti navedena).
- Koordinate: **pikseli originalne slike 640x640**, `x1 < x2`, `y1 < y2` (isto kao anotacije DeepPCB-a).
- `cls`: **1-6** kao u DeepPCB-u (1 open, 2 short, 3 mousebite, 4 spur, 5 copper, 6 pin-hole).
  YOLO/RT-DETR koriste 0-5, pa pri upisu dodati 1. Faster R-CNN (torchvision) ima 0 = pozadina i oznake 1-6,
  tj. 7 izlaznih klasa, što se poklapa s ovim formatom.
- `score`: pouzdanost 0-1. Za klasični pristup bez pouzdanosti upisati 1.0.
- Eksperiment 1 bez klasifikatora (samo lokalizacija) upisuje `"cls": 0`; evaluacija ga tada računa kao detekciju bez klase.

## Evaluacija (`src/evaluation/`)
- Isti skupovi za sve: validacija iz `src/data/to_yolo.py` (seed 42) za odluke, test skup samo za konačno poređenje.
- Metrike: precision, recall, F1, mAP@0.5, mAP@0.5:0.95 i **mAP/F1 pri IoU 0.33** (prag iz izvornog rada DeepPCB).
- IoU se računa na pikselnim boxovima; jedan GT box može se upariti s najviše jednom predikcijom.
- Matrica konfuzije i metrike po klasi.

## Vrijeme inferencije
Mjeriti na istom hardveru za sve modele: batch 1, sama inferencija (bez učitavanja slike), prosjek preko 100 slika nakon zagrijavanja. Upisati hardver (npr. "Colab T4") uz rezultat.

## Seedovi
Duboki modeli se treniraju s bar 2-3 seeda; prijavljuje se prosjek i odstupanje.

## Tabela rezultata: `results/results.csv`
Kolone: `experiment, variant, seed, split, mAP50, mAP50_95, mAP33, precision, recall, f1, ms_per_image, hardware, commit`

## Alati za evaluaciju (`src/evaluation/`)

Prije prvog korištenja: `python -m src.evaluation.check_fake` (provjera na lažnim predikcijama; mora ispisati `SVE PROVJERE PROŠLE`).

| Šta | Komanda |
|---|---|
| Metrike jednog modela | `python -m src.evaluation.evaluate --pred results/<exp>/predictions_val.json --split val` |
| Matrica konfuzije, zamjene klasa, recall po veličini, slike s greškama | `python -m src.evaluation.analysis --pred ... --split val --out results/<exp>/analysis` |
| Upis reda u `results/results.csv` | `python -m src.evaluation.compare add --experiment exp3 --variant yolo_template --seed 0 --pred ... --split val --ms 12.4 --hardware "Colab T4"` |
| Tabela poređenja i grafovi | `python -m src.evaluation.compare table --split val --out results` |
| Robusnost | `run_robustness(predict, experiment, variant)` iz `src/evaluation/robustness.py` (vidi docstring), zatim `python -m src.evaluation.robustness --csv results/robustness_*.csv` |

Napomene:
- Precision, recall i F1 u `results.csv` su pri pragu pouzdanosti s najboljim F1 (IoU 0.5).
- AP se računa kao površina ispod cijele PR krive, pa se mAP može malo razlikovati od onog koji javlja Ultralytics (101 tačka); u izvještaju koristiti isključivo vrijednosti iz ovog modula.
- Vrijeme inferencije: `measure_inference_time` iz `compare.py` (batch 1, 10 zagrijavanja, prosjek 100 slika, isti hardver za sve modele).
- Funkcija za robusnost prima `(test_img, temp_img, image_id)` kao `uint8` nizove (640, 640), sive tonove, i vraća `(boxes, scores, cls)`.
- Neispravne predikcije **prekidaju** evaluaciju s porukom o grešci: klasa izvan 0-6, pomiješane klase 0 i 1-6, `score` izvan 0-1 ili NaN, box s `x1 >= x2` ili `y1 >= y2`. Slike bez zapisa u predikcijama su samo upozorenje (računaju se kao bez detekcija).
- Tumačenje pomaka u robusnosti: modeli koji ne koriste template (YOLO baseline, Faster R-CNN, RT-DETR) služe kao kontrola, jer bi pomak slike trebao slabo utjecati na njih. Za metode koje porede test i template (klasični pristup, YOLO + template) pad metrika mjeri otpornost na **neporavnanje** test i template slike, a ne na pomjeranje objekata.
