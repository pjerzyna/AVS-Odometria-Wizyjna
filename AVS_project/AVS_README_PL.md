# Vectorized Motion Positioning Component (MPC) for UAV Visual Odometry

Zaawansowany, w pełni zwektoryzowany potok monokularowej odometrii wizyjnej (Visual Odometry - VO), działający jako komponent pozycjonowania ruchu (MPC) dla bezzałogowych statków powietrznych (UAV). System przetwarza sekwencje obrazów lotniczych w celu ciągłej estymacji wektora stanu i rekonstrukcji dwuwymiarowej trajektorii lotu platformy.

---

## 1. O projekcie i zestawie danych

### Kontekst teoretyczny i odrzucenie KLT Optical Flow
Tradycyjne podejścia do odometrii wizyjnej w robotyce latającej często opierają się na rzadkim śledzeniu przepływu optycznego, np. metodą **Kanade-Lucas-Tomasi (KLT) Optical Flow**. Choć algorytmy te są wydajne obliczeniowo, wykazują krytyczne podatności w warunkach lotniczych: są wysoce wrażliwe na gwałtowne zmiany orientacji (obroty wokół osi yaw/pitch/roll), chwilowe rozmycia obrazu (motion blur) oraz okresowe zaniki tekstury podłoża. 

W tym projekcie **świadomie zrezygnowano z metod śledzenia Optical Flow na rzecz potoku opartego na jawnej detekcji punktów kluczowych i dopasowaniu niezmienniczych deskryptorów binarnych**. Taka architektura zapewnia:
* **Odporność na rotację:** Deskryptory są stabilne niezależnie od zmian kursu UAV.
* **Weryfikowalność geometryczną:** Każde dopasowanie przechodzi rygorystyczny filtr RANSAC.
* **Możliwość łatwego rozbudowania:** Architektura pozwala w przyszłości na implementację pętli zamknięcia (Loop Closure) za pomocą mechanizmów Bag-of-Words (BoW), co jest niemożliwe przy czystym Optical Flow.

### Powiązanie z badaniami naukowymi
Wykorzystywany w projekcie zestaw danych lotniczych **Village Flyover** (składający się z 62 klatek sekwencji planarnej) pochodzi bezpośrednio z materiałów badawczych powiązanych z niedawną rozprawą doktorską **dr. inż. Tomasza Pogorzelskiego** (*"System samolokalizacji wizyjnej dla latających platform bezzałogowych"*, Politechnika Warszawska, 2025). 

Badania naukowe dr. Pogorzelskiego koncentrują się na problematyce autonomicznej nawigacji UAV w środowiskach pozbawionych sygnału GPS (GPS-denied environments), gdzie odometria wizyjna stanowi kluczowy element fuzji sensorycznej. Niniejsza implementacja stanowi bezpośrednie, inżynierskie rozwinięcie oraz weryfikację założeń stabilności geometrycznej opisanych w tej pracy doktorskiej, w szczególności w zakresie radzenia sobie z anomaliami śledzenia nad trudnym, powtarzalnym terenem (takim jak korony lasów czy struktury rolnicze).

---

## 2. Architektura Algorytmu (Pipeline)

Dla każdej kolejnej pary klatek ($\mathcal{I}_{t-1}, \mathcal{I}_t$) potok wykonuje zestaw w pełni zwektoryzowanych operacji NumPy, eliminując kosztowne pętle `for` w języku Python:

```
  [Klatka I_t] ---> Detekcja FAST ---> Harris Scoring ---> Grid NMS ---> Intensity Centroid ---> rBRIEF Descriptor
                                                                                                        |
  [Klatka I_t-1] -> [Analogiczny proces detekcji i opisu deskryptorami rBRIEF] -------------------------> |
                                                                                                        v
  Trajektoria <--- Ekstrakcja (tx, ty) <--- Historyczny Buffer (CVM) / Safeguards <--- RANSAC Homography <--- Popcount LUT Matching
```

1.  **Detekcja FAST (FAST detection):** Izoluje punkty kandydackie za pomocą testu segmentu z więzem ciągłości $N \ge 9$ na okręgu 16 pikseli. Czułość sterowana jest parametrem `--threshold`.
2.  **Ocena punktów metodą Harrisa (Harris corner scoring):** Punkty FAST są filtrowane przez globalnie zmapowany tensor strukturalny drugich pochodnych obrazu. Wyliczany jest jawny współczynnik odpowiedzi narożnikowej:
    $$R = (I_{xx}I_{yy} - I_{xy}^2) - 0.04 \cdot (I_{xx} + I_{yy})^2$$
    Odrzucane są punkty leżące na krawędziach ($R < 0$) oraz w obszarach jednorodnych ($R  pprox 0$).
3.  **Tłumienie niemaksymalne (Grid-cell NMS):** Obraz jest dzielony na wirtualną siatkę o promieniu $r = 16	ext{ px}$. Punkty są mapowane wektorowo na unikalne identyfikatory komórek: `Cell_ID = (y // r) * 100000 + (x // r)`. Z każdego koszyka redukcja NumPy wybiera wyłącznie jeden punkt o najwyższej miarze Harrisa, co daje złożoność $O(N)$ i gwarantuje równomierne pokrycie kadru.
4.  **Orientacja względem centroidu intensywności:** Wewnątrz patcha 31×31 wyliczane są momenty geometryczne $m_{pq} = \sum x^p y^q \mathcal{I}(x,y)$, z których wyznaczany jest dominujący kąt struktury $	heta = 	ext{atan2}(m_{01}, m_{10})$.
5.  **Rotacyjno-niezmienniczy deskryptor rBRIEF:** Zamiast losowego losowania par punktów, system wczytuje 256 zoptymalizowanych par testów binarnych z pliku `orb_descriptor_positions.txt` (wyselekcjonowanych metodami uczenia maszynowego w celu minimalizacji korelacji). Współrzędne te są w locie przekształcane 3D tensorem rotacji $R_	heta$ na wygładzonym obrazie Gaussa $\mathcal{G}$:
    $$	ext{bit}_k = \mathcal{G}(p + R_	heta \mathbf{u}_k) < \mathcal{G}(p + R_	heta \mathbf{v}_k)$$
    Wynik pakowany jest w 4 słowa `uint64` (256-bitowy ciąg binarny).
6.  **Zwektoryzowany Matcher popcount LUT:** Odległość Hamminga wyliczana jest bez pętli za pomocą operacji `np.bitwise_xor` na rozgłoszonych macierzach (broadcasting) o kształtach `(N, 1)` i `(1, M)`. Sumowanie bitów realizowane jest błyskawicznie poprzez podział słów 64-bitowych na 16-bitowe sekcje i mapowanie przez tablicę prawdy `_LUT16`. Jednoznaczność dopasowań weryfikuje bezwzględny test stosunku marginesu:
    $$(d_1 \le 	ext{max\_hamming}) \land (d_1 \le d_2 - 	ext{margin})$$
7.  **Estymacja Homografii + RANSAC:** Algorytm `cv2.findHomography` z filtrem geometrycznym RANSAC odrzuca punkty błędnie dopasowane (outliers) na podstawie jednostronnego błędu reprojekcji forward w układzie współrzędnych jednorodnych:
    $$d^2 = \| \mathbf{x}_t - \hat{H}\mathbf{x}_{t-1} \|^2$$
8.  **Weryfikacja Safeguards i Ekstrakcja Ruchu:** Jeśli liczba poprawnych inlierów spełnia kryterium gęstości ($|\mathcal{M}_{	ext{inliers}}| \ge N_{	ext{min}}$), relatywne translacje kamery $(t_x, t_y)$ są wyciągane bezpośrednio z elementów macierzy homografii $H_{0,2}$ i $H_{1,2}$.

---

## 3. Mechanizmy Odpornościowe (Robustness Safeguards)

W celu zabezpieczenia estymatora stanu przed eksplozją błędu i załamaniem trajektorii nad trudnym geometrycznie terenem (np. klatki 41--51 nad monotonnym kompleksem leśnym) oraz anomaliami sekwencyjnymi danych (cykliczne zgubione klatki co 11 klatek), wdrożono wielopoziomowe zabezpieczenia:

### Bramkowanie Rozmiaru Konsensusu (Consensus Size Gating)
Estymacja homografii z małej próby wejściowej drastycznie podnosi podatność na przetrenowanie geometryczne RANSAC. Wprowadzono sztywne ograniczenie dolne na wielkość zbioru wsparcia ($N_{	ext{min}} = 30$). Jeśli liczba inlierów spadnie poniżej progu, model geometryczny zostaje odrzucony, zapobiegając integracji zniekształconych, nieliniowych skoków translacji.

### Historyczny Model Stałej Prędkości (Constant Velocity Model - CVM)
W przypadku wykrycia utraty trackingu lub odrzucenia homografii przez bramkowanie, system nie zamraża pozycji i nie pozwala na skok trajektorii. Aktywowane jest **kinematyczne coastowanie (kinematic coasting)**. System wyznacza średni wektor przemieszczenia na podstawie kroczącego bufora historii prędkości (rolling history buffer) z ostatnich $K=3$ udanie zarejestrowanych klatek:
$$\mathbf{t}_t = rac{1}{K}\sum_{i=1}^{K} \mathbf{t}_{t-i}$$
Pozwala to na płynne, fizycznie poprawne "przeperycypowanie" drona nad obszarami bezteksturowymi, eliminując skoki pozycji.

### Nieliniowe Clampowanie Kinematyczne (Translation Clamping)
Z racji wysokiej częstotliwości próbkowania kamery względem fizycznych możliwości dynamicznych UAV, przemieszczenie ramka-do-ramki jest ograniczone nieliniowym stożkiem prędkości:

$$
\mathbf{t}_{\text{clamped}} = \mathbf{t}_t \quad \text{jeśli} \quad \|\mathbf{t}_t\| \le \Delta_{\text{max}}
$$

$$
\mathbf{t}_{\text{clamped}} = \Delta_{\text{max}} \frac{\mathbf{t}_t}{\|\mathbf{t}_t\|} \quad \text{w przeciwnym wypadku}
$$

Próg $\Delta_{	ext{max}} = 300	ext{ px}$ skutecznie odcina anomalie wywołane chwilowym błędnym dopasowaniem powtarzalnych tekstur rolniczych, utrzymując ciągłość wektora stanu.

---

## 4. Kluczowe Parametry Uruchomieniowe

| Flaga / Parametr | Wartość Domyślna | Opis Funkcjonalny |
|------------------|------------------|-------------------|
| `--threshold`    | `20`             | Progowanie czułości detektora FAST. Niższa wartość zwiększa gęstość punktów w słabo oświetlonych strefach. |
| `--n_best`       | `500`            | Górny limit punktów kluczowych na klatkę po selekcji miarą Harrisa i redukcji NMS. |
| `--n_matches`    | `150`            | Liczba najlepszych par deskryptorów przekazywanych z modułu dopasowania na wejście RANSAC. |
| `--ransac_thr`   | `5.0`            | Maksymalny błąd reprojekcji forward w pikselach, definiujący przynależność punktu do inlierów. |
| `--save_strips`  | `False`          | Flaga debugowania – generuje graficzne pasy dopasowań (side-by-side) w folderze wyjściowym. |

---

## 5. Wyniki Eksperymentalne

Wykonanie potoku z aktywnymi zabezpieczeniami na sekwencji *Village Flyover* pozwoliło na wygenerowanie kompletnych logów telemetrycznych i bezbłędne zamknięcie pełnej pętli estymacji trajektorii:

| Kategoria Metryki | Wartość Empiryczna |
|-------------------|--------------------|
| Całkowita liczba przetworzonych klatek | 62 |
| Średnia liczba inlierów konsensusu na parę | **64.1** |
| Maksymalna zaobserwowana liczba inlierów (Klatka 10) | 238 |
| Minimalna liczba inlierów (Siatka utraty sygnału) | 0 (Bezpieczne przejście CVM) |
| Skumulowana długość wyznaczonej trajektorii | **14131.2 px** |
| Końcowa estymowana pozycja UAV $(X, Y)$ | **(+865.4, +13380.6) px** |

Wdrożony model **Constant Velocity Model** pomyślnie zinterpolował pozycję platformy podczas awarii odczytu na klatkach 11, 22, 33 oraz nad gęstym kompleksem leśnym na klatce 50, zapewniając pełną zbieżność estymatora bez dryfu katastrofalnego. Wygenerowany wykres trajektorii lotu (`trajectory.png`) charakteryzuje się gładką, fizyczną krzywizną, odporną na szum pomiarowy.