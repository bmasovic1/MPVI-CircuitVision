# Plan rada

## Eksperimenti i nosioci

| # | Eksperiment | Šta ispituje | Nosilac | Folder |
|---|---|---|---|---|
| 1 | OpenCV + Random Forest/SVM | koliko se može postići klasičnom obradom slike i klasifikacijom izdvojenih regija | Član 1 | `src/experiment_1_classical/` |
| 2 | Faster R-CNN | dvostepena detekcija objekata (torchvision, 6 klasa + pozadina) | Član 2 | `src/experiment_2_faster_rcnn/` |
| 3 | YOLO baseline + YOLO + template | doprinos referentne slike, uz kontrolisano poređenje i ablaciju | Berin Mašović | `src/experiment_3_yolo_template/` |
| 4 | RT-DETR | transformer-based detekcija objekata | Član 4 | `src/experiment_4_rtdetr/` |

**Zajedničku evaluaciju (`src/evaluation/`) piše Berin Mašović.**

Svaki nosilac eksperimenta priprema adapter koji predikcije svog modela zapisuje u formatu iz [`FORMAT.md`](FORMAT.md).

| Dio evaluacije | Fajl | Ko |
|---|---|---|
| IoU i uparivanje predikcija s ground truthom (jedan GT box najviše jednoj predikciji) | `metrics.py` | Berin Mašović |
| Precision, recall, F1, mAP@0.5, mAP@0.5:0.95, mAP/F1 pri IoU 0.33 | `metrics.py` | Berin Mašović |
| Matrica konfuzije i metrike po klasi | `analysis.py` | Berin Mašović |
| Mjerenje vremena inferencije, upis u `results/results.csv`, tabela poređenja | `compare.py` | Berin Mašović |
| Robusnost (šum, kontrast, pomak testne slike u odnosu na template) | `robustness.py` | Berin Mašović |


## Zajednički protokol

1. Prije treniranja dogovoriti fiksne train/validation/test podjele, format anotacija i format predikcija.
2. Validacijski skup koristiti za izbor hiperparametara i odluke tokom razvoja; test skup ne koristiti za tuning, nego samo za konačno poređenje.
3. Svi modeli moraju predavati predikcije u formatu iz [`FORMAT.md`](FORMAT.md), u pikselnim koordinatama originalne slike 640×640.
4. Prijaviti precision, recall, F1, mAP@0.5, mAP@0.5:0.95 i metrike pri IoU 0.33 radi poređenja s izvornim DeepPCB radom.
5. Vrijeme inferencije mjeriti na istom hardveru, uz isti batch size i isti protokol zagrijavanja.
6. Dokumentovati konfiguraciju, seed, trajanje treniranja, hardver, rezultate i analizu grešaka za svaku varijantu.
7. **Napomena o podjeli podataka:** službeni `test` skup dijeli 7 od svojih 11 grupa s `trainval` (provjereno na repou), pa test slike dolaze s istih grupa ploča kao dio trening slika. Naša validacija (`split_by_group`) je odvojena po grupama (2 grupe, 248 slika), pa može davati nešto pesimističnije brojke od testa, a zbog samo dvije grupe je i prilično nepouzdana. Oboje treba navesti u izvještaju pri tumačenju rezultata.

## Redoslijed rada

| Faza | Zajednički cilj | Rezultat koji se vidi u repou |
|---|---|---|
| 1. Zajednička osnova | Validacija anotacija, EDA, fiksiranje splitova i formata predikcija; evaluaciju piše Berin, a ostali u međuvremenu pripremaju svoj loader | `src/evaluation/`, `notebooks/01_eda`, dokumentovan split |
| 2. Prve verzije | Svaki član implementira minimalnu verziju svog eksperimenta;| prvi rezultati u `results/`, unosi u `docs/experiments.md` |
| 3. Poboljšanja | Template varijanta i ablacija; optimizacija Faster R-CNN, RT-DETR i klasičnog pristupa | poredbene tabele, grafovi, analiza grešaka |
| 4. Završna evaluacija | Konačno poređenje na test skupu, robusnost, demo, poster i izvještaj | finalni `results.csv`, `demo/`, poster i izvještaj |

Faze se mogu preklapati: nakon što se usaglasi format predikcija, članovi mogu paralelno razvijati svoje eksperimente. Ne treba čekati da jedan model bude potpuno završen prije početka ostalih.

## Istraživačka pitanja

1. Koliko se defekata može lokalizovati klasičnom obradom slike i koliko ih klasifikator ispravno razvrsta?
2. Kako se dvostepeni detektor Faster R-CNN poredi s ostalim pristupima po preciznosti, odzivu i brzini?
3. **Glavno pitanje:** poboljšava li korištenje referentne template slike detekciju u odnosu na YOLO koji koristi samo testnu sliku?
4. Kako se RT-DETR ponaša na malim PCB defektima u poređenju s ostalim detektorima?
5. Koliko su pristupi otporni na šum, promjenu kontrasta i pomak testne slike u odnosu na template sliku?

## Robusnost

Robusnost ispitati na kontrolisanim izmjenama ulaza, uz očuvanje originalnih anotacija gdje transformacija ne mijenja geometriju objekata:
- šum i blur;
- promjena kontrasta/svjetline;
- relativni pomak testne slike u odnosu na template sliku.

Relativni pomak posebno je važan za eksperimente koji koriste razliku test/template: metode koje koriste samo testnu sliku ne bi trebale biti direktno pogođene pomakom templatea, dok metode koje porede slike mogu izgubiti poravnanje. Bilježiti veličinu pomaka u pikselima i mjeriti promjenu metrika u odnosu na neizmijenjeni ulaz.

## Upravljanje rizikom i resursima

Sva četiri eksperimenta su planirani ciljevi. RT-DETR i Faster R-CNN mogu zahtijevati više GPU vremena, zato rano napraviti kratki probni trening i zabilježiti memoriju, vrijeme po epohi i ostvariv broj epoha. Ako je Colab ograničen, prvo smanjiti nepotrebne eksperimente hiperparametara i koristiti razumno podešene pretrained težine; jasno dokumentovati razlike u konfiguraciji. Ne donositi zaključak da je model lošiji ako nije dobio uporedivu priliku za treniranje. Ako vremena ili GPU-a ne bude, RT-DETR prvi otpada; eksperimenti 1-3 sami daju cjelovitu priču i odgovor na glavno pitanje.

## Podjela odgovornosti

- **Član 1:** implementacija i eksperimentalna analiza OpenCV + SVM/Random Forest pristupa.
- **Član 2:** implementacija i eksperimentalna analiza Faster R-CNN pristupa, uključujući vlastiti dataset loader.
- **Berin Mašović:** YOLO baseline, YOLO + template i ablacija; čuva kontrolisane uslove poređenja; piše zajedničku evaluaciju i kod za robusnost.
- **Član 4:** implementacija i eksperimentalna analiza RT-DETR pristupa.
- **Svi nosioci eksperimenata:** adapter koji predikcije svog modela zapisuje u zajedničkom formatu i pokretanje evaluacije i robusnosti nad svojim modelom.

Svi članovi dodatno učestvuju u pregledu rezultata, dokumentaciji, vizualizacijama, posteru, zaključcima i pripremi za usmeni ispit. Svaki član treba imati vidljiv tehnički doprinos kroz kod, eksperimente i analizu, a ne samo dokumentaciju.