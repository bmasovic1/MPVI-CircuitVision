# Definicija problema i pregled literature

## 1. Uvod

Štampane ploče (PCB) su osnova gotovo svakog elektronskog uređaja, a defekti nastali u proizvodnji (prekinuti ili
spojeni vodovi, višak ili manjak bakra) mogu uzrokovati kvar cijelog uređaja. Zato se proizvedene ploče provjeravaju,
tradicionalno vizuelnim pregledom, a sve češće automatskom optičkom inspekcijom (AOI) i metodama dubokog učenja.
Težina zadatka je u tome što su defekti vrlo sitni u odnosu na cijelu ploču, a neki tipovi se međusobno vizuelno slični.

Ovaj projekat uspoređuje četiri pristupa automatskoj detekciji i klasifikaciji PCB defekata na javnom skupu DeepPCB:
klasičnu obradu slike s klasifikatorom, dvostepeni detektor (Faster R-CNN), jednostepeni detektor (YOLO) u dvije varijante
(bez i s referentnom slikom) i transformer detektor (RT-DETR). Glavno pitanje je da li referentna (template) slika
ispravne ploče poboljšava detekciju u odnosu na pristup koji koristi samo sliku s defektima.

## 2. Definicija problema

**Zadatak:** detekcija objekata, tj. lokalizacija svakog defekta (bounding box) i određivanje njegovog tipa.

**Ulaz:** slika PCB-a 640×640 piksela i, za pristupe koji to koriste, ispravna template slika iste ploče.

**Izlaz:** za svaki detektovani defekt box (x1, y1, x2, y2), tip (jedan od šest) i pouzdanost.

**Tipovi defekata:** `open` (prekid voda), `short` (kratki spoj), `mousebite` (ugriz na vodu), `spur` (izdanak),
`copper` (višak bakra), `pin-hole` (rupica).

**Mjera uspjeha:** precision, recall, F1 i mAP (pri IoU 0.5, 0.5:0.95 i 0.33), po klasama i ukupno, uz matricu konfuzije i
vrijeme inferencije. Sve se računa jedinstvenim kodom opisanim u [`EVALUACIJA.md`](EVALUACIJA.md).

## 3. Dataset

Koristi se **DeepPCB** (Tang i sur. [1]): 1500 parova slika 640×640, svaki par čine slika s defektima (`_test`) i ispravna referentna slika (`_temp`).
Službena podjela: 1000 slika za trening i validaciju, 500 za test. Slike su binarizirane (crno bakar, bijelo podloga), a defekti su većinom umjetno dodani.
Potpun opis (nastanak, format, raspodjele, kvalitet podataka, ograničenja) je u [`DATASET.md`](DATASET.md); ovdje su samo ključni nalazi.

| Klasa | Trening + validacija (defekata) | Test (defekata) | Prosj. veličina boxa (px) |
|---|---|---|---|
| open | 1283 | 659 | 45 × 38 |
| short | 1028 | 478 | 45 × 40 |
| mousebite | 1379 | 586 | 36 × 32 |
| spur | 1142 | 483 | 35 × 33 |
| copper | 1010 | 464 | 38 × 38 |
| pin-hole | 1031 | 470 | 35 × 34 |
| **Ukupno** | **6873** | **3140** | |

**Ključni nalazi (izmjereno na skupu):**
- Defekti su sitni: srednja strana boxa je oko 37 px na slici od 640 px, a 98% defekata zauzima manje od 1% površine slike.
- Klase su relativno uravnotežene, ali skoro svaka slika sadrži skoro sve klase (prosječno 5 od 6), što ukazuje na umjetno umetanje.
- Defekata po slici ima 1 do 15 (README navodi 3 do 12).
- 85% svih razlika između test i template slike leži izvan označenih defekata (rubovi i šum), a kod oko 9% parova postoji pomak od nekoliko piksela.

**Ograničenja skupa, bitna za tumačenje rezultata:**
- Defekti su većinom umjetni, pa rezultati ne moraju vrijediti za prave proizvodne ploče. Drugi autori navode i da je DeepPCB crno-bijel,
  dok neki drugi skupovi koriste slike u boji, što otežava prenos modela [4].
- Test skup većinom (66% slika) dolazi iz grupa ploča kojih nema u treningu, pa mjeri generalizaciju na nove ploče. Naša validacija odvojena
  je po grupama, ali sadrži samo 2 grupe (248 slika) i zato je prilično nepouzdana.
- Izvorni rad javlja mAP 98.6% pri 62 FPS [1], pa je skup blizu "riješenog" i razlike između modela mogu biti male.
- Jedna slika u test skupu (`44000020_test.jpg`) ima ucrtane okvire i nazive klasa u samim pikselima slike.

## 4. Relevantnost

PCB je sastavni dio gotovo svakog elektronskog proizvoda, a njegov kvalitet određuje pouzdanost proizvoda [5]. Otkrivanje defekata
utiče na prinos proizvodnje i pouzdanost kasnijih faza sklapanja [6]. Rane metode zasnovane na razlici slika i klasičnoj obradi
mogu zakazati zbog neuračunatih oblika defekata i osjetljivosti na parametre [1], pa se istraživanje usmjerilo na duboko učenje.
Ono traži anotirane podatke, a javnih skupova je malo [1, 2, 6], što otežava poređenje metoda. Zbog toga je poređenje
više pristupa pod istim uslovima (isti podaci, isti protokol, isti kod za evaluaciju) korisno i izvan same ocjene projekta.

## 5. Pregled literature

**Skupovi podataka i izvorni model.** Tang i sur. [1] objavljuju DeepPCB i detektor koji postiže mAP 98.6% pri 62 FPS. Huang i Wei [2]
objavljuju drugi javni skup sa šest vrsta sintetizovanih defekata (PKU-Market-PCB). Ove dvije radove najčešće koriste kasniji autori.

**Dvostepeni detektori.** TDD-Net [3] gradi na Faster R-CNN [8], uz feature pyramid, k-means anchore i online hard example mining,
a javlja mAP oko 98.9%. Prije citiranja provjeriti u radu na kojem je skupu to izmjereno (vjerovatno PKU, ne DeepPCB).

**Jednostepeni detektori (YOLO).** YOLO [7] je danas najčešći izbor. YOLO-pdd [4] kombinuje YOLOv5 s višerazinskim modulima sa
hijerarhijskim rezidualnim vezama radi bolje detekcije sitnih defekata i otpornosti na šum. GESC-YOLO [5] mijenja YOLOv8n da bi model
bio lagan, cilja defekte manje od 1% površine slike i javlja mAP do 99%.

**Transformer detektori.** RT-DETR (Zhao i sur. [9]) izbacuje anchore i potiskivanje nemaksimuma. MRC-DETR [6] ga prilagođava za PCB
i navodi da se RT-DETR na PCB defektima suočava s nedovoljnim izdvajanjem karakteristika i velikom računskom složenošću pri
spajanju razmjera, te da javnih skupova za neopremljene ploče nema dovoljno.

**Upotreba referentne slike.** U navedenim radovima koje smo pregledali detektori se uglavnom oslanjaju na sliku s defektima, a ne na
direktnu razliku prema ispravnoj ploči. To nismo sistematski provjerili kroz cijelu literaturu, pa tvrdnju o praznini treba uzeti oprezno.

## 6. Praznina i doprinos projekta

Većina pregledanih radova poboljšava arhitekturu (laganiji ili precizniji YOLO) na jednom ulazu. Mi umjesto nove arhitekture ispitujemo:
1. doprinos referentne slike u kontrolisanom poređenju istog modela sa i bez nje, uz ablaciju ulaza;
2. poređenje četiri porodice pristupa pod istim protokolom i istim kodom za evaluaciju;
3. otpornost na šum, promjenu kontrasta i neporavnanje test i template slike;
4. analizu grešaka po klasama i po veličini defekta.

## 7. Istraživačka pitanja

1. Koliko se defekata može lokalizovati klasičnom obradom slike i koliko ih klasifikator ispravno razvrsta?
2. Kako se dvostepeni detektor Faster R-CNN poredi s ostalim pristupima po preciznosti, odzivu i brzini?
3. **Glavno pitanje:** poboljšava li referentna template slika detekciju u odnosu na YOLO koji koristi samo testnu sliku?
4. Kako se RT-DETR ponaša na sitnim defektima u poređenju s ostalim detektorima?
5. Koliko su pristupi otporni na šum, promjenu kontrasta i pomak testne slike u odnosu na template?

## 8. Literatura

1. S. Tang, F. He, X. Huang, J. Yang, *Online PCB Defect Detector On A New PCB Defect Dataset*, arXiv:1902.06197, 2019.
2. W. Huang, P. Wei, *A PCB Dataset for Defects Detection and Classification*, arXiv:1901.08204, 2019.
3. R. Ding, L. Dai, G. Li, H. Liu, *TDD-net: a tiny defect detection network for printed circuit boards*, CAAI Transactions on Intelligence Technology, 2019, doi:10.1049/trit.2019.0019.
4. B. Liu, D. Chen, X. Qi, *YOLO-pdd: A Novel Multi-scale PCB Defect Detection Method Using Deep Representations with Sequential Images*, arXiv:2407.15427, 2024.
5. X. Kong, G. Liu, Y. Gao, *GESC-YOLO: Improved Lightweight Printed Circuit Board Defect Detection Based Algorithm*, Sensors 25(10), 3052, 2025, doi:10.3390/s25103052.
6. J. Cao, H. Wu, X. Zhang, L. Tan, H. Zhang, *MRC-DETR: An Adaptive Multi-Residual Coupled Transformer for Bare Board PCB Defect Detection*, arXiv:2507.03386, 2025.
7. J. Redmon, S. Divvala, R. Girshick, A. Farhadi, *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016.
8. S. Ren, K. He, R. Girshick, J. Sun, *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NeurIPS 2015.
9. Y. Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection*, CVPR 2024 (arXiv:2304.08069).
10. G. Jocher et al., *Ultralytics YOLOv8*, https://github.com/ultralytics/ultralytics

**Status provjere referenci:** reference 1, 2, 4, 5 i 6 provjerene su (naslov, autori, godina, izvor). Referenca 3: naslov i izdavač provjereni,
autore i DOI još jednom uporediti s originalnim PDF-om. Reference 7-10 navedene su po sjećanju i **nisu provjerene**; prije predaje
provjeriti autore, godinu i izvor (posebno 9). Brojke iz radova (npr. 98.9% za TDD-Net, 99% za GESC-YOLO) citirati tek nakon provjere u originalnom radu.
