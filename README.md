# CircuitVision

Detekcija i klasifikacija defekata na štampanim pločama (PCB) – projekat iz predmeta MPVI 2026/27.

**Tim:** Mašović Berin, Prašović Asmir, Džindo Nedim, Yahia Elhattab Aissa

## Cilj projekta

Implementirati i eksperimentalno uporediti četiri različita pristupa za detekciju i klasifikaciju šest tipova PCB defekata:
`open`, `short`, `mousebite`, `spur`, `copper` i `pin-hole`.

Glavno istraživačko pitanje je da li korištenje referentne (template) slike može poboljšati detekciju defekata u odnosu na pristup koji koristi samo testnu sliku.

## Trenutni status

- [x] Skelet repoa, učitavanje DeepPCB-a, podjela po grupama, konverzija u YOLO format
- [ ] Zajednička evaluacija (`src/evaluation/`): IoU 0.5, 0.33, P/R/F1 i mAP
- [ ] EDA (distribucija klasa, veličine boxova i pregled anotacija)
- [ ] Eksperiment 1: OpenCV + Random Forest/SVM
- [ ] Eksperiment 2: Faster R-CNN
- [ ] Eksperiment 3: YOLO baseline + YOLO s template informacijama + ablacija
- [ ] Eksperiment 4: RT-DETR
- [ ] Robusnost, demo, poster i izvještaj

Plan rada, istraživačka pitanja i nosioci: [`docs/PLAN.md`](docs/PLAN.md) · Format predikcija: [`docs/FORMAT.md`](docs/FORMAT.md) · Dnevnik eksperimenata: [`docs/experiments.md`](docs/experiments.md)

## Problem i dataset

**Ulaz:** slika PCB-a i, za pristupe koji to koriste, odgovarajuća ispravna template slika.  
**Izlaz:** bounding box, tip i pouzdanost za svaki detektovani defekt.  
**Metrike:** precision, recall, F1, mAP i vrijeme inferencije.

Koristi se [DeepPCB](https://github.com/tangsanli5201/DeepPCB), skup od 1.500 parova test/template slika rezolucije 640×640. Defekti su većinom umjetno dodani, što treba uzeti u obzir pri interpretaciji rezultata.

Potpun opis problema i pregled literature: [`docs/PROBLEM.md`](docs/PROBLEM.md).

## Četiri eksperimenta

1. **OpenCV + SVM/Random Forest** – oduzimanje testne i template slike, obrada razlike, izdvajanje kandidata za defekte i klasifikacija pronađenih regija.
2. **Faster R-CNN** – dvostepeni detektor objekata implementiran pomoću torchvision.
3. **YOLO + template** – kontrolisano poređenje običnog YOLO baseline modela s varijantom koja koristi testnu sliku, template sliku i njihovu razliku. Ablacijom se ispituje doprinos pojedinačnih ulaza.
4. **RT-DETR** – transformer-based detektor objekata.


## Pokretanje

```bash
git clone https://github.com/bmasovic1/MPVI-CircuitVision
cd MPVI-CircuitVision
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py                        # preuzima DeepPCB u data/
python -m src.data.to_yolo                             # pravi data/yolo (train/val/test)
```

## Struktura

```text
src/data/                         učitavanje, validacija i konverzija podataka
src/experiment_1_classical/       OpenCV oduzimanje + Random Forest/SVM
src/experiment_2_faster_rcnn/     Faster R-CNN (torchvision)
src/experiment_3_yolo_template/   YOLO baseline, YOLO + template, ablacija
src/experiment_4_rtdetr/          RT-DETR
src/evaluation/                   zajedničke metrike, robusnost i poređenje
results/                          predikcije i results.csv (format: docs/FORMAT.md)
scripts/                          pomoćne skripte
notebooks/                        Colab i EDA notebookovi
docs/                             problem, plan, format i dnevnik eksperimenata
demo/                             demo aplikacija/skripta
```

## Dokumentacija i dnevnik rada

- Plan i podjela posla: [`docs/PLAN.md`](docs/PLAN.md)
- Dnevnik eksperimenata: [`docs/experiments.md`](docs/experiments.md)
- Zajednički dokument tima: https://docs.google.com/document/d/1ndZP65o3jClo5vU07Cjei9QkXWC-h4XX1Xx2fHTHnPo/edit?tab=t.0


## Podjela rada

Konačna imena nosilaca upisati nakon dogovora unutar tima. Svaki nosilac eksperimenta prilagođava svoj model zajedničkom formatu predikcija iz [`docs/FORMAT.md`](docs/FORMAT.md).

| Član | Zadatak | Folder |
|---|---|---|
| Član 1 | Eksperiment 1: OpenCV + Random Forest/SVM | `src/experiment_1_classical/` |
| Član 2 | Eksperiment 2: Faster R-CNN | `src/experiment_2_faster_rcnn/` |
| Berin Mašović | Eksperiment 3: YOLO baseline + YOLO + template + ablacija; zajednička evaluacija | `src/experiment_3_yolo_template/`, `src/evaluation/` |
| Član 4 | Eksperiment 4: RT-DETR | `src/experiment_4_rtdetr/` |

## Literatura

1. S. Tang et al., *Online PCB Defect Detector On A New PCB Defect Dataset*, arXiv:1902.06197, 2019.
2. W. Huang, P. Wei, *A PCB Dataset for Defects Detection and Classification*, arXiv:1901.08204, 2019.
3. R. Ding et al., *TDD-net: a tiny defect detection network for printed circuit boards*, CAAI Trans. Intell. Technol., 2019.

Dataset je namijenjen istraživačkoj upotrebi; prije redistribucije podataka provjeriti uslove licence na izvornoj stranici DeepPCB-a.
