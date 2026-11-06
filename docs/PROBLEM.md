# Definicija problema

## Zadatak
**Ulaz:** slika PCB-a (640x640, grayscale) i, opcionalno, ispravna template slika iste ploče.
**Izlaz:** za svaki defekt bounding box, tip (open, short, mousebite, spur, copper, pin-hole) i pouzdanost.
**Mjera uspjeha:** precision, recall, F1 i mAP (IoU 0.5 i 0.33) po klasi i ukupno; matrica konfuzije.
Radi se o detekciji objekata, a ne samo klasifikaciji slike: defekt treba i locirati.

## Zašto je težak
Defekti su sitni (najmanja strana boxa oko 20 px na slici od 640 px, prosjek 35-47 px), a razlike su fine
(npr. *mousebite* i *spur* su slični po obliku i veličini).

## Relevantnost
PCB je osnova gotovo svakog elektronskog uređaja, a proizvodni defekti mogu uzrokovati kvar cijelog uređaja.
Ručna kontrola je spora i podložna umoru, pa se koristi automatska optička inspekcija.
Rane metode zasnovane na razlici slika i klasičnoj obradi mogu zakazati zbog neuračunatih obrazaca
defekata i osjetljivosti na hiperparametre [1]. Duboko učenje je robusnije, ali traži anotirane podatke,
a javnih skupova je malo, što otežava poređenje metoda [1, 2].
_(TODO: dodati jedan izvor o značaju/industrijskoj inspekciji PCB-a.)_

## Pregled literature
- **Dataset i referentni model:** Tang i sur. [1] objavljuju DeepPCB (1500 parova slika, 6 tipova defekata)
  i detektor s mAP 98.6% pri 62 FPS. Defekti su većinom umjetno dodani, što je ograničenje.
- **Dvostepeni detektori:** TDD-Net [3] (Faster R-CNN, feature pyramid, k-means anchori, online hard example
  mining) postiže ~98.9% mAP; izvještavano na drugom skupu (PKU), ne na DeepPCB. _(Provjeriti u radu prije citiranja.)_
- **Jednostepeni detektori (YOLO):** radovi poput YOLO-pdd [4] i GESC-YOLO [5] ciljaju sitne defekte i
  lagane modele. Rad Adibhatla i sur. koristi YOLOv5 samo za razlikovanje defektnih od ispravnih ciljeva.

## Praznina i doprinos projekta
Većina radova koristi samo sliku s defektima. Mi ispitujemo doprinos template slike, poredimo klasični
pristup i duboke detektore pod istim uslovima, te analiziramo greške po klasama (vidi `docs/PLAN.md`).

## Literatura
1. S. Tang, F. He, X. Huang, J. Yang, *Online PCB Defect Detector On A New PCB Defect Dataset*, arXiv:1902.06197, 2019.
2. W. Huang, P. Wei, *A PCB Dataset for Defects Detection and Classification*, arXiv:1901.08204, 2019.
3. R. Ding, L. Dai, G. Li, H. Liu, *TDD-net: a tiny defect detection network for printed circuit boards*, CAAI Trans. Intell. Technol., 2019, doi:10.1049/trit.2019.0019.
4. *YOLO-pdd: A Novel Multi-scale PCB Defect Detection Method Using Deep Representations with Sequential Images*, arXiv:2407.15427. _(dodati autore)_
5. *GESC-YOLO: Improved Lightweight Printed Circuit Board Defect Detection Based Algorithm*. _(dodati autore, godinu, izvor)_
6. J. Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016.
7. G. Jocher et al., *Ultralytics YOLOv8*, https://github.com/ultralytics/ultralytics
