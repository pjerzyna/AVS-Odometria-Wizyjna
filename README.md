## O projekcie i zestawie danych

Projekt polega na stworzeniu własnego skryptu realizującego tzw. **odometrię wizyjną**. Zadanie polega na estymacji ruchu kamery na podstawie analizy sekwencji obrazów. 

COŚ W KIERUNKU OPTICAL FLOW ROBIMY, CZY NIE? WSPOMINAMY COŚ O TYM, IDZIEMY W TYM KIERUNKU?

Wykorzystywany w projekcie zestaw danych pochodzi z materiałów, na których podczas pisania swojej pracy doktorskiej pracował Tomasz Pogorzelski.

COŚ TAM JESZCZE OPISAŁBYM O AUTORZE, ABY BYŁO SKŁADNIE WSZYSTKO ROZPISANE I SENSOWNIE ODNOSZĄCE SIĘ DO TEMATYKI NASZEGO PROJEKTU

---

### Wykorzystanie zoptymalizowanych par punktów (ORB/BRIEF)

W projekcie zaimplementowano deskryptor **BRIEF** (jako część algorytmu ORB). Zamiast każdorazowego, losowego generowania par punktów do testów binarnych, zdecydowano się na użycie predefiniowanego zestawu współrzędnych z pliku `orb_descriptor_positions.txt`. Użycie stałej bazy punktów gwarantuje, że ten sam punkt kluczowy zawsze wygeneruje identyczny deskryptor, co umożliwia skuteczne porównywanie obrazów między różnymi uruchomieniami programu.

Współrzędne w pliku nie są losowe – jest to zestaw 256 par zoptymalizowanych za pomocą algorytmu uczenia maszynowego (zgodnie ze specyfikacją ORB). Wybrane pary niosą najwięcej unikalnych informacji i nie dublują się wzajemnie. Współrzędne te (zdefiniowane pierwotnie dla orientacji $0^\circ$) są w locie dynamicznie obracane o kąt $\theta$ (wyznaczony z centroidu intensywności), co zapewnia deskryptorowi odporność na obroty kamery.

---

## Algorithm pipeline

### Dla każdej pary klatek (Per frame-pair):

1. **Detekcja FAST (FAST detection)** Wyszukuje potencjalne narożniki za pomocą testu segmentu (*Accelerated Segment Test*). Proces jest kontrolowany przez parametr `--threshold` (domyślnie `20`; niższa wartość = więcej wykrytych punktów).

2. **Ocena punktów metodą Harrisa (Harris corner scoring)** Każdy punkt wykryty przez algorytm FAST jest ponownie oceniany miarą odpowiedzi Harrisa. Pozwala to na wybranie stabilnych geometrycznie narożników i odrzucenie punktów leżących na krawędziach.

3. **Tłumienie niemaksymalne (NMS - Non-Maximum Suppression)** Zachowuje tylko najsilniejszy punkt w otoczeniu o promieniu `nms_radius` pikseli. Zapewnia to równomierne rozłożenie punktów kluczowych na obrazie i eliminuje redundantne skupiska.

4. **Orientacja względem centroidu intensywności (Intensity-centroid orientation)** Oblicza dominujący kierunek dla każdego punktu kluczowego na podstawie fragmentu obrazu (*patcha*) o wymiarach 31×31 pikseli za pomocą wzoru:
   theta = atan2(m01, m10)

5. **BRIEF odporny na obroty (rBRIEF - Rotation-invariant BRIEF)** Wczytuje 256 zoptymalizowanych par $(x1, y1, x2, y2)$ z pliku `orb_descriptor_positions.txt`, obraca je o wyznaczony kąt $\theta$ i wykonuje 256 binarnych testów intensywności. Wynikiem jest kompaktowy, 256-bitowy deskryptor (zapisywany w pamięci jako 4 zmienne typu `uint64`).

6. **Dopasowanie Brute-Force Hamming + test stosunku Lowe'a (Brute-force Hamming matching + Lowe's ratio test)** Wyszukuje dwa najbliższe sąsiedztwa dla każdego deskryptora przy użyciu odległości Hamminga. Filtruje dopasowania, pozostawiając tylko te najbardziej jednoznaczne, które spełniają warunek: `dist1 / dist2 < 0.75`.

7. **Estymacja homografii + odrzucanie punktów odstających RANSAC (Homography estimation + RANSAC outlier rejection)** Wykorzystuje funkcję `cv2.findHomography(..., cv2.RANSAC, reprojection_threshold=5.0)`. Eliminuje ona błędne dopasowania geometryczne (*outliers*) i pozostawia wyłącznie poprawne, spójne przestrzennie pary punktów (*inliers*).

8. **Ekstrakcja translacji (Translation extraction)** Odczytuje ostateczne wartości przesunięcia klatki `tx = H[0,2]` oraz `ty = H[1,2]` bezpośrednio z wyznaczonej macierzy homografii $H$.

---

## Kluczowe parametry

| Flaga / Parametr | Domyślnie | Działanie / Opis |
|------------------|-----------|------------------|
| `--threshold`    | `20`      | Czułość algorytmu FAST - niższa wartość pozwala wykryć więcej narożników. |
| `--n_best`       | `500`     | Maksymalna liczba najlepszych punktów kluczowych zachowywanych dla klatki po filtracji Harris + NMS. |
| `--n_matches`    | `150`     | Liczba najlepszych dopasowań przekazywanych jako wejście do algorytmu RANSAC. |
| `--ransac_thr`   | `5.0`     | Maksymalny dopuszczalny błąd reprojekcji dla punktów zgodnych (*inliers*) w RANSAC [w pikselach]. |
| `--save_strips`  | `off`     | Flaga włączająca zapisywanie obrazu PNG z liniami dopasowań dla każdej kolejnej pary klatek. |