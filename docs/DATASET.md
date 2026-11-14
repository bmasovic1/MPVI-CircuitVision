# DeepPCB: opis skupa podataka

Skup koji koristimo je **DeepPCB** (Tang i sur., 2019): <https://github.com/tangsanli5201/DeepPCB>. Ovaj dokument opisuje šta u njemu zaista
ima. Svi brojevi su izračunati skriptom `python -m src.data.eda` (traje oko minutu) i mogu se ponovo provjeriti; slike su u
`docs/figures/`, a sve statistike u `docs/figures/eda_stats.json`. Gdje je nešto pretpostavka, a ne izmjereno, to piše.

## 1. Kako je skup nastao (prema autorima)
- Slike dolaze s linijskog CCD skenera, rezolucije oko **48 piksela po milimetru**. Originalne slike šablona i testiranih ploča su
  veličine oko 16 000 × 16 000 piksela, a ispravni šabloni su ručno provjereni i očišćeni.
- Zatim su izrezane na isječke od **640 × 640** piksela i **poravnate tehnikom template matching**. Primijenjena je i **binarizacija** s pažljivo
  odabranim pragom da se izbjegnu smetnje zbog osvjetljenja. Autori navode da su registracija slika i prag uobičajeni koraci za precizno lociranje PCB defekata.
- Na pravim testiranim slikama ima malo defekata, pa su autori **ručno dodali umjetne defekte** prema obrascima stvarnih defekata, što daje, prema njihovom
  opisu, oko 3 do 12 defekata po slici. Svaki defekt ima box i oznaku klase.
- Fizičke dimenzije: slika od 640 px odgovara oko **13.3 mm** ploče, a prosječan defekt (strana oko 37 px) oko **0.76 mm**.
  *(Izračunato iz podatka od 48 px/mm koji navode autori.)*

## 2. Struktura i format
```
PCBData/
  group00041/                      grupa (jedan od 11 prefiksa u skupu)
    00041/00041000_test.jpg        slika s defektima
    00041/00041000_temp.jpg        ispravna template slika
    00041_not/00041000.txt         anotacije: x1 y1 x2 y2 tip  (odvojeno razmakom, ne zarezom kako piše README)
  trainval.txt, test.txt           službena podjela (putanje u split fajlovima završavaju na NNN.jpg, a ne na NNN_test.jpg)
```
Klase: 1 open, 2 short, 3 mousebite, 4 spur, 5 copper, 6 pin-hole (0 se ne koristi). Box je (x1, y1) gornji lijevi i (x2, y2) donji desni ugao u pikselima.
Službeni README ne objašnjava šta je "grupa". Pretpostavljamo da grupa odgovara jednoj ploči ili seriji, ali to nije potvrđeno.

## 3. Veličina i podjela
| | Slika | Defekata |
|---|---|---|
| Trening + validacija (`trainval.txt`) | 1000 | 6873 |
| Test (`test.txt`) | 500 | 3140 |
| **Ukupno** | **1500** | **10013** |
| Naš trening (iz trainval) | 752 | 5150 |
| Naša validacija (iz trainval, po grupama) | 248 | 1723 |

Parovi slika po grupama (slika `07_grupe.png`):

| Grupa | trainval | test | ukupno | defekata |
|---|---|---|---|---|
| group00041 | 200 | 21 | 221 | 1642 |
| group12000 | 0 | 14 | 14 | 93 |
| group12100 | 0 | 146 | 146 | 919 |
| group12300 | 0 | 98 | 98 | 600 |
| group13000 | 198 | 18 | 216 | 1451 |
| group20085 | 291 | 34 | 325 | 2428 |
| group44000 | 50 | 50 | 100 | 747 |
| group50600 | 50 | 29 | 79 | 432 |
| group77000 | 100 | 7 | 107 | 558 |
| group90100 | 0 | 74 | 74 | 483 |
| group92000 | 111 | 9 | 120 | 660 |

Ima 1501 template fajlova za 1500 parova: jedan template (`90100034_temp.jpg`) nema odgovarajuću test sliku ni anotaciju i ne koristi se.
Od 1500 korištenih template slika 1498 su različite (dva duplikata), a duplikata test slika nema.

## 4. Klase defekata
Kako izgledaju u ovom skupu, prema primjerima na slici `05_primjeri_po_klasi.png` (uz svaki primjer lijevo test, desno template). Na slikama je **crno bakar, a bijelo podloga**:

| Klasa | Šta je | Kako izgleda |
|---|---|---|
| open | prekid voda | bijeli procjep prekida crni vod |
| short | kratki spoj | crni most spaja dva susjedna voda |
| mousebite | ugriz | mali urez ili neravnina na rubu voda |
| spur | izdanak | mali crni izdanak koji viri iz voda |
| copper | višak bakra | izolovana crna mrlja na bijeloj podlozi |
| pin-hole | rupica | mala bijela rupa unutar crne površine |

Raspodjela defekata (slika `01_raspodjela_klasa.png`):

| Klasa | Naš trening | Naša validacija | Test | Slika s klasom (trainval) |
|---|---|---|---|---|
| open | 1003 (19.5%) | 280 (16.3%) | 659 (21.0%) | 86.2% |
| short | 830 (16.1%) | 198 (11.5%) | 478 (15.2%) | 73.3% |
| mousebite | 1000 (19.4%) | 379 (22.0%) | 586 (18.7%) | 85.6% |
| spur | 848 (16.5%) | 294 (17.1%) | 483 (15.4%) | 81.1% |
| copper | 730 (14.2%) | 280 (16.3%) | 464 (14.8%) | 83.8% |
| pin-hole | 739 (14.3%) | 292 (16.9%) | 470 (15.0%) | 87.7% |

Klase su relativno uravnotežene (u trainval od 1010 do 1379 defekata po klasi). Gotovo svaka slika sadrži skoro sve klase:
prosječno **4.98 od 6** različitih klasa po slici. To je znak umjetnog umetanja defekata, a ne prirodne raspodjele.
Validacija se po raspodjeli klasa primjetno razlikuje od treninga (npr. `short` 11.5% naspram 16.1%), jer se sastoji od samo dvije grupe.

## 5. Veličina i položaj defekata
Defekti su **sitni**: srednja strana boxa je 36.7 px (medijan 34.4), 5. do 95. percentil 28 – 54 px, najmanji 20 px, najveći 129 px.
**98.2%** defekata zauzima manje od 1% površine slike, a **32.0%** ima stranu manju od 32 px
(po uobičajenoj COCO podjeli to su "mali" objekti, površine ispod 32² px; ostali su uglavnom "srednji").

| Klasa | prosj. širina × visina (px) | medijan strane (px) | 5.–95. percentil | najveća | omjer š/v (medijan) |
|---|---|---|---|---|---|
| open | 45 × 38 | 36 | 29 – 65 | 129 | 1.13 |
| short | 45 × 40 | 39 | 31 – 60 | 122 | 1.19 |
| mousebite | 36 × 32 | 32 | 26 – 44 | 91 | 1.14 |
| spur | 35 × 33 | 33 | 28 – 44 | 74 | 1.07 |
| copper | 38 × 38 | 36 | 28 – 51 | 80 | 1.00 |
| pin-hole | 35 × 34 | 33 | 26 – 45 | 121 | 1.03 |

`open` i `short` su u prosjeku veći (41 i 41 px), a `mousebite` i `spur` najmanji (33 i 34 px) (slika `03_velicine_boxova.png`).
Položaj defekata je raspoređen po cijeloj slici (slika `04_prostorni_raspored.png`).

**Broj defekata po slici** (slika `02_defekata_po_slici.png`): prosjek 6.87 u trainval i 6.28 u testu, medijan 7 i 6.
Mjereno, raspon je **1 do 15** u trainval (2 do 13 u testu), što je šire od "3 do 12" iz README-a.

## 6. Kakve su slike
- Sve slike su 640 × 640. Slike su **binarne**: 98.14% piksela je unutar 10 nivoa od čiste crne ili bijele (ostatak su JPEG artefakti na rubovima).
- Udio tamnih piksela jako varira: prosjek 0.35, od 0.00 (gotovo prazna slika) do 0.86.
  Ima slika s malo strukture (kod 10 od 1500 parova gotovo da nema ivica).
- **Tehnički detalj za učitavanje:** većina slika je u sivom modu (`L`), ali ih ima 38 test i 24 template u modu `RGB` (tri kanala). Sve osim jedne imaju gotovo iste vrijednosti u
  sva tri kanala (razlika ispod 30 nivoa). Učitavanje treba normalizovati (`Image.open(...).convert("L")`), inače se oblici nizova razlikuju.
- **Jedna slika ima ucrtane anotacije:** `44000020_test.jpg` (u službenom **test** skupu) ima u pikselima slike zelene okvire i bijele oznake s nazivima klasa
  (obojeno je oko 1.5% piksela). Njen template je čist. Model bi na toj slici u principu mogao "čitati" oznake. Utjecaj je najviše 1 od 500 test slika.
  Preporuka: ne brisati je iz testa (poređenje s literaturom), ali znati za nju i, ako rezultati budu granični, provjeriti ih i bez nje.
  Ostale slike nismo ručno pregledali; obojene piksele smo tražili automatski, pa bi eventualno ucrtane sive oznake ostale neotkrivene.

## 7. Odnos test i template slike
Slika `06_primjer_para.png` prikazuje par: test s anotacijama, template i razliku. Razlika sadrži defekte, ali i **tanke konture oko gotovo svih oblika**.
Izmjereno na svih 1500 parova (prag razlike 100 od 255):
- **84.7% svih različitih piksela leži izvan označenih boxova** (medijan po slici 79%).
  Dakle, većina razlika između test i template slike nisu označeni defekti, nego rubovi i šum (binarizacija, JPEG, mala neporavnanja).
- **Procjena neporavnanja** (fazna korelacija, na 1490 parova s dovoljno strukture): kod **139 parova (9.3%)**
  pomak template slike za nekoliko piksela smanjuje razliku za bar 20%; tipičan pomak je 3 px (od 1 do 10 px). Kod ostalih pomak ne pomaže.
  Medijan udjela različitih piksela prije i poslije pomjeranja je gotovo isti (1.402% i 1.358%).
  To upućuje na to da je većina parova **poravnata**, a manji dio ima pomak od nekoliko piksela. Konture u razlici kod poravnatih parova potiču iz
  nečeg drugog (vjerovatno binarizacije i kompresije); to nismo posebno ispitali.

## 8. Provjera kvaliteta podataka
| Provjera | Rezultat |
|---|---|
| Svi parovi i anotacije postoje | da (1000 + 500) |
| Neispravni boxovi (van slike ili nulte veličine) | 0 |
| Boxovi uz sam rub slike (unutar 2 px) | 65 |
| Parovi preklopljenih boxova unutar iste slike | 167 (defekti se ponekad dodiruju) |
| Duplikati test slika | 0 |
| Template bez test slike | 1 (`90100034_temp.jpg`) |
| Broj defekata po slici prema README-u / izmjereno | 3 – 12 / 1 – 15 |
| Slike u RGB modu | 62 (38 test, 24 template), jedna s ucrtanim anotacijama |

## 9. Podjela koju koristimo i njena ograničenja
- **Službena podjela:** 1000 trening/validacija i 500 test. Trainval dolazi iz 7 grupa, test iz 11.
- **Test i trening dijele grupe, ali samo djelimično:** 7 grupa je i u trainval i u testu, ali te grupe čine samo **168 od 500 test slika (34%)**.
  Preostalih **332 test slika (66%)** dolazi iz 4 grupa koje u treningu uopšte ne postoje (12000, 12100, 12300, 90100).
  Test dakle većinom mjeri generalizaciju na ploče koje model nije vidio. *(Ovo ispravlja ranije pojednostavljeno tvrđenje da test "dijeli grupe s treningom".)*
- **Naša validacija** izdvaja se iz trainval po grupama (`split_by_group`, seed 42) i sastoji se od samo **2 grupe** (13000, 44000, 248 slika).
  Dobra strana je što nema istih ploča u treningu i validaciji; loša što je mala i drugačije raspoređena po klasama, pa je procjena nepouzdana.
  Obje validacijske grupe se pojavljuju i u testu.

## 10. Šta to znači za naše eksperimente
| Nalaz | Posljedica |
|---|---|
| 85% razlika test/template nisu defekti | Eksperiment 1: čisto oduzimanje daje puno lažnih kandidata; treba morfologija, filtriranje po veličini i tolerancija na poravnanje. |
| 32.0% defekata ispod 32 px, skoro svi ispod 1% površine | Detektori (Eksperimenti 2-4) imaju problem s malim objektima; vrijedi isprobati veću ulaznu rezoluciju ili isječke (hipoteza, nije testirano). |
| Oko 9.3% parova ima pomak od nekoliko piksela | Eksperimenti koji koriste template (1 i 3) trebaju biti otporni na malo neporavnanje; to ispituje test robusnosti. |
| Test većinom iz nepoznatih grupa | Rezultati na testu odražavaju generalizaciju; razlika između validacije i testa može biti velika u oba smjera. |
| Skoro svaka slika ima sve klase | Raspodjela je umjetna; rezultati ne moraju vrijediti za prave ploče, što treba napisati u zaključku. |
| Slike u RGB modu i jedna s ucrtanim oznakama | Loader mora normalizovati mod slike; napomenuti slučaj `44000020_test.jpg`. |
| Isti transform na test i template | Ako se koristi augmentacija (flip, rotacija) za modele s template slikom, ista transformacija mora se primijeniti na oba. |

## 11. Šta nije utvrđeno
- Šta tačno znači "grupa" i da li različite slike iste grupe pokrivaju isti dio ploče.
- Uzrok tankih kontura u razlici kod poravnatih parova.
- Da li ima ucrtanih sivih oznaka na drugim slikama (tražene su samo obojene).
- Da li umjetni defekti odgovaraju stvarnim defektima po izgledu i raspodjeli.
