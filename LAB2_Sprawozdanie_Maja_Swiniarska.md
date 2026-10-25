

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy

**Zajęcia nr / temat:** 2. Sieci rekurencyjne: budowa i uczenie

**Autor / grupa:** Maja Świniarska GL.04

**Data:** 22.10.2026

## 1. Cel ćwiczenia

Sprawdzenie, jak liczba jednostek ukrytych (hidden), długość okna (OKNO) i typ komórki (RNN, LSTM, GRU) wpływają na jakość prognozy kolejnej próbki szeregu czasowego (test MSE), w porównaniu z heurystyką „prognoza = ostatnia próbka okna”.

## 2. Metoda i implementacja

Narzędzia: PyTorch, NumPy, Matplotlib; CPU w Colabie. 

Dane: sin(t) + 0.1·sin(3t) + szum N(0; 0,05²), 2000 punktów, generowane raz. Wariancja szumu 0,0025 to dolna granica MSE.

Model: jedna warstwa RNN/LSTM/GRU + Linear(hidden, 1) na ostatnim stanie (many-to-one). Adam, MSE, clipping gradientu 5,0.

Podział: chronologiczny 80/20 na oknach; zmiana OKNO ponownie tworzy okna i podział.

Warunki: epoki=20, lr=1e-2, batch=64, ziarna 0/1/2; wynik = średnia ± odch. std. z mean(hist[-5:]).

Eksperymenty: 

A — hidden=8/16/32/64 (GRU, okno 30); 

B — OKNO=10/30/60/120 (GRU, h=32); C — RNN/LSTM/GRU (h=32, okna 10 i 120).

Hipoteza A: MSE spadnie przy hidden 8→32, powyżej 32 zysk będzie pomijalny, bez przeuczenia.

Hipoteza B: okno 10 będzie najgorsze (bliskie heurystyce), najlepsze 30–60, okno 120 bez zysku względem 60 i z dłuższym czasem.

## 3. Wyniki

**Punkt odniesienia (RNN, h=32, okno 30, 30 epok):** test MSE = 0,00377 (ostatnia epoka), 0,00386 (średnia z 5 ostatnich). Heurystyka „ostatnia próbka”: 0,00631.

EKSPERYMENT A  (GRU, okno 30)
| Konfiguracja | Parametry | Test MSE (śr. ± odch.) | Czas [s] |
|---|---|---|---|
| *Heurystyka „ostatnia próbka”* | — | *0,00631* | — |
| GRU, h=8 | 273 | 0,00380 ± 0,00004 | 5,7 |
| GRU, h=16 | 929 | 0,00374 ± 0,00002 | 6,6 |
| GRU, h=32 | 3393 | 0,00376 ± 0,00001 | 8,2 |
| GRU, h=64 | 12929 | 0,00375 ± 0,00008 | 13,8 |

<img width="889" height="390" alt="image" src="https://github.com/user-attachments/assets/5e330a01-8936-4918-91b5-f03ef9a8cc19" />


EKSPERYMENT B (GRU, h=32)
| Konfiguracja | Parametry | Test MSE (śr. ± odch.) | Czas [s] |
|---|---|---|---|
| *Heurystyka, okno 10* | — | *0,00628* | — |
| *Heurystyka, okno 30* | — | *0,00631* | — |
| *Heurystyka, okno 60* | — | *0,00633* | — |
| *Heurystyka, okno 120* | — | *0,00631* | — |
| GRU, okno 10 | 3393 | 0,00414 ± 0,00019 | 3,2 |
| GRU, okno 30 | 3393 | 0,00376 ± 0,00001 | 8,2 |
| GRU, okno 60 | 3393 | **0,00343 ± 0,00021** | 14,3 |
| GRU, okno 120 | 3393 | 0,00344 ± 0,00019 | 28,2 |

<img width="889" height="390" alt="image" src="https://github.com/user-attachments/assets/1930d520-44df-4b17-ab51-63ced8bea6e0" />


EKSPERYMENT C (h=32)
| Konfiguracja | Parametry | Test MSE (śr. ± odch.) | Czas [s] |
|---|---|---|---|
| *Heurystyka, okno 10* | — | *0,00628* | — |
| *Heurystyka, okno 120* | — | *0,00631* | — |
| RNN, okno 10 | 1153 | 0,00417 ± 0,00025 | 1,5 |
| LSTM, okno 10 | 4513 | 0,00415 ± 0,00011 | 3,0 |
| GRU, okno 10 | 3393 | 0,00414 ± 0,00019 | 3,2 |
| RNN, okno 120 | 1153 | 0,00400 ± 0,00001 | 10,1 |
| LSTM, okno 120 | 4513 | 0,00375 ± 0,00013 | 18,2 |
| GRU, okno 120 | 3393 | **0,00344 ± 0,00019** | 28,2 |

<img width="889" height="390" alt="image" src="https://github.com/user-attachments/assets/1b143f5e-b7f4-4b04-ab7f-b6f3a4a512ab" />


Najlepsza konfiguracja: GRU, h=32, okno 60 (0,00343 ± 0,00021; 54% błędu heurystyki).

<img width="790" height="370" alt="image" src="https://github.com/user-attachments/assets/e98b5810-6477-42dc-8a29-b24d93499e2c" />

<img width="847" height="336" alt="image" src="https://github.com/user-attachments/assets/405f1895-33db-484e-a500-3883bf72ff93" />



## 4. Analiza i interpretacja

A: różnice między hidden (do 0,00006) mieszczą się w rozrzucie ziaren; już h=8 wystarcza, a większy model tylko wydłuża trening.

B: okno 10 (<8% okresu ≈126 próbek) daje za mało informacji o fazie; od okna 60 (≈pół okresu) poprawy brak, a czas rośnie ~proporcjonalnie do okna. To interpretacja, nie osobno testowana.

C: przy oknie 10 typy są nieodróżnialne; przy oknie 120 kolejność GRU < LSTM < RNN. Zysk z okna 10→120: RNN ok. 4%, LSTM ok. 10%, GRU ok. 17%. Modele mają różną liczbę parametrów.

Diagnostyka: test/trening 0,98–1,03 (brak przeuczenia), brak nan; krzywe poszarpane przy lr=1e-2 (największy skok straty testowej 0,00257).

Ograniczenia: konfiguracje porównywane na zbiorze testowym (brak walidacyjnego); różne okna → nieco inne zbiory testowe; 3 ziarna; jeden syntetyczny sygnał.

## 5. Wnioski
1. Większa liczba jednostek nie poprawia wyniku: dla h=8–64 MSE ≈ 0,0037–0,0038 (różnice w granicach rozrzutu), a czas rośnie.

2. Najlepsze okno to 60 (MSE 0,00343); okno 120 daje to samo przy dwukrotnie dłuższym treningu, okna 10 i 30 są gorsze. Prawdopodobnie okno 60 zawiera już dość informacji o fazie sygnału.

3. RNN i komórki bramkowane reagują różnie: wydłużenie okna 10→120 daje RNN ok. 4%, LSTM ok. 10%, GRU ok. 17% poprawy. RNN się nie pogorszył.

4. Wszystkie modele są wyraźnie lepsze od heurystyki (54–66% jej MSE), ale nie osiągają granicy szumu 0,0025.

5. Nie zaobserwowano przeuczenia (test/trening 0,98–1,03) ani poważnej niestabilności, tylko poszarpane krzywe. Wyraźnego niedouczenia też nie widać (na krzywej najlepszej konfiguracji strata przestaje spadać przed końcem 20 epok), choć błąd pozostaje powyżej poziomu szumu 0,0025.

6. Hipotezy: A potwierdzona w części (plateau, brak przeuczenia; poprawy 8→32 nie było). B w większości (okno 10 najgorsze, 120 bez zysku, czas rośnie), ale okno 10 nie było bliskie heurystyce, a okno 60 było wyraźnie lepsze od 30.


---


