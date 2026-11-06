# Podaci

Dataset **DeepPCB** se ne nalazi u ovom repozitoriju (134 MB, licenca samo za istraživačke svrhe).
Izvor: https://github.com/tangsanli5201/DeepPCB

## Preuzimanje

```
python scripts/download_data.py
```

Skripta klonira dataset u `data/DeepPCB/` i provjerava da su svi fajlovi na mjestu
(1000 trening + 500 test parova slika).

## Struktura nakon preuzimanja

```
data/
  DeepPCB/PCBData/group*/NNNNN/NNNNNNNN_test.jpg   slika s defektima
                         NNNNN/NNNNNNNN_temp.jpg   ispravna template slika
                         NNNNN_not/NNNNNNNN.txt    anotacije
  yolo/                                            generisano: python -m src.data.to_yolo
```

## Format anotacija
Svaki red: `x1 y1 x2 y2 tip` (odvojeno razmakom). Tipovi: 1 open, 2 short, 3 mousebite, 4 spur, 5 copper, 6 pin-hole.
