# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy

**Zajęcia nr / temat:** 4. Obsługa długich sekwencji: SimpleRNN, LSTM, GRU i Conv1D

**Autor / grupa:** Maja Świniarska GL.04

**Data:** 05.11.2026

## 1. Cel ćwiczenia

Porównanie SimpleRNN, LSTM, GRU i Conv1D w zadaniu sumowania wartości wskazywanych z opóźnieniem (adding problem, znacznik w kroku `t` wskazuje wartość z kroku `t − 8`) dla długości sekwencji L = 50 i L = 100. Dodatkowo zbadano wpływ rozmiaru stanu ukrytego SimpleRNN oraz dylatacji (pola recepcyjnego) Conv1D.

## 2. Metoda i implementacja

**Narzędzia:** Python 3.13.16, PyTorch 2.11.0+cu130 (modele nn.RNN, nn.LSTM, nn.GRU, nn.Conv1d, optymalizator Adam), NumPy (generator danych), pandas (tabele wyników), Matplotlib (wykresy), Jupyter/Colab (środowisko). Urządzenie: GPU (CUDA, Tesla T4).

**Dane.** Wejście `[L, 2]`: kanał 1 — liczby `U[0,1]`, kanał 2 — znaczniki `0/1`. Dwie różne pozycje znaczników losowane z kroków 9…L (od 1); znacznik w kroku `t` wskazuje wartość z kroku `t − 8`. 

Cel: suma dwóch wskazanych wartości.
Kontrola generatora: znaczniki w krokach 9 i 12 -> wskazane kroki 1 i 4, wartości 0,8506 i 0,2698, suma 1,1204 = cel z generatora. Dodatkowo test pętlą na 500 przykładach (dokładnie 2 znaczniki, pozycje ≥ 9, cel = suma) zakończony powodzeniem. Przykład losowy z generatora: znaczniki w krokach 15 i 17 -> wskazane kroki 7 i 9, wartości 0,8229 i 0,2492, suma 1,0722 = cel.

**Modele**

| Model | Konfiguracja | Parametry |
|---|---|---|
| SimpleRNN / LSTM / GRU | 1 warstwa, stan 64, odczyt z ostatniego stanu + `Linear` | 4 417 / 17 473 / 13 121 |
| Conv1D | 3 warstwy, 64 kanały, jądro 3, krok 1, dylatacje [1,2,4], padding = dylatacja, `mean` po czasie + `Linear` | 25 217 |
| Wariant SimpleRNN | stan ukryty 128 | 17 025 |
| Wariant Conv1D | dylatacje [1,1,1] (reszta bez zmian) | 25 217 |

**Konfiguracja** Zbiory (dla każdego L, generowane raz i wspólne dla wszystkich modeli): train 4000, test 800; ziarno 0 (dane: `SEED+100+L` i `SEED+200+L`). Adam, lr 5·10⁻³, 30 epok, partia 128, strata MSE, clipping normy gradientu 1,0. Walidacji nie użyto - ustawień nie dobierano na podstawie wyników, zbiór testowy służy tylko do końcowej oceny. Czas mierzony tylko dla pętli uczenia (`perf_counter` + `cuda.synchronize()`, po rozgrzewce). Baseline: stała predykcja równa średniej celu z treningu.

## 3. Wyniki

**Tabela 1.** Porównanie modeli (test MSE; ostatnia kolumna - stosunek do stałej predykcji).

| L | Model | Parametry | Czas uczenia [s] | Train MSE | Test MSE | Test MSE / stała |
|---|---|---|---|---|---|---|
| 50 | SimpleRNN | 4 417 | 2,5 | 0,1748 | 0,1734 | 1,041 |
| 50 | LSTM | 17 473 | 2,3 | 0,1744 | 0,1716 | 1,030 |
| 50 | GRU | 13 121 | 2,3 | 0,0823 | **0,0893** | **0,536** |
| 50 | Conv1D | 25 217 | 2,4 | 0,1607 | 0,1603 | 0,963 |
| 50 | Stała predykcja | – | – | – | 0,1666 | 1,000 |
| 100 | SimpleRNN | 4 417 | 2,7 | 0,1611 | 0,1569 | 0,998 |
| 100 | LSTM | 17 473 | 2,6 | 0,1615 | 0,1570 | 0,999 |
| 100 | GRU | 13 121 | 2,4 | 0,1590 | 0,1543 | 0,981 |
| 100 | Conv1D | 25 217 | 2,2 | 0,1570 | 0,1542 | 0,981 |
| 100 | Stała predykcja | – | – | – | 0,1572 | 1,000 |

<img width="786" height="490" alt="image" src="https://github.com/user-attachments/assets/2a46ed8b-e6ba-4cf3-9eb4-2125c2850b9a" />




**Tabela 2.** Zmiana jednego elementu, L = 100 (te same dane i ustawienia).

| Zmiana | Parametry | Czas [s] | Test MSE | Δ test MSE | Parametry / podst. | Czas / podst. |
|---|---|---|---|---|---|---|
| SimpleRNN (stan 64, podstawowy) | 4 417 | 2,7 | 0,1569 | – | 1,00 | 1,00 |
| SimpleRNN (stan 128) | 17 025 | 2,1 | 0,1713 | +0,0144 | 3,85 | 0,81 |
| Conv1D (dyl. [1,2,4], podstawowy) | 25 217 | 2,2 | 0,1542 | – | 1,00 | 1,00 |
| Conv1D (dyl. [1,1,1]) | 25 217 | 2,3 | 0,1543 | +0,0001 | 1,00 | 1,01 |

Test MSE stałej predykcji dla L = 100: 0,1572.

<img width="1489" height="390" alt="image" src="https://github.com/user-attachments/assets/6578af7f-3282-4e94-b822-18e9b24617f2" />


**Tabela 3.** Pole recepcyjne Conv1D (`R = 1 + (k−1)·Σd`, k = 3).

| Dylatacje | R (wzór) | R (gradient) | Max odległość kroków w polu (R−1) | Obejmuje znacznik i próbkę (odl. 8)? |
|---|---|---|---|---|
| [1, 2, 4] | 15 | 15 | 14 | tak |
| [1, 1, 1] | 7 | 7 | 6 | nie |

**Przykład z dużym błędem** (Conv1D, L = 100, przykład testowy nr 192).

| Znacznik w kroku | Wskazuje krok | Wartość v | Odległość wskazanego kroku od końca |
|---|---|---|---|
| 45 | 37 | 0,0322 | 63 |
| 77 | 69 | 0,0766 | 31 |

Suma prawidłowa: 0,1089 · predykcja: 1,0635 · błąd: +0,9547 · błąd kwadratowy: 0,9114.

<img width="1089" height="370" alt="image" src="https://github.com/user-attachments/assets/0131ea19-3e96-433f-a7c4-571006509c2b" />


## 4. Analiza i interpretacja

**Względem baseline'u.** Stała predykcja ma test MSE 0,1666 (L = 50) i 0,1572 (L = 100); wartości różnią się, bo zbiory testowe są różne. Przy 800 próbkach błąd standardowy test MSE wynosi ok. 0,007 (oszacowanie dla stałej predykcji), więc różnice rzędu kilku tysięcznych nie są dowodem nauczenia się zadania.
- **L = 50:** tylko GRU wyraźnie pokonał stałą predykcję (0,0893, czyli 54% błędu stałej mniej); jego train MSE 0,0823 pokazuje, że nauczył się zadania częściowo. SimpleRNN (1,041) i LSTM (1,030) są na poziomie stałej lub nieco powyżej, Conv1D (0,963) poniżej o ok. jeden błąd standardowy, więc bez wiarygodnej poprawy.
- **L = 100:** wszystkie cztery modele mają stosunek do stałej 0,981–0,999, czyli w granicach szumu. Żaden nie pokonał stałej predykcji w sposób wiarygodny.

  <img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/be2a33eb-b6e6-40af-b53f-2df8abe6b2c6" />


**Wpływ długości (L = 50 → 100).** Błąd GRU wzrósł z 0,0893 do 0,1543, a przewaga nad stałą zniknęła (0,536 → 0,981). Dla pozostałych modeli spadek bezwzględnego MSE (np. SimpleRNN 0,1734 → 0,1569) wynika ze spadku MSE stałej predykcji, a nie z poprawy modelu — stosunek do stałej się nie zmienił. Zgadza się to z oczekiwaniem, że dłuższa sekwencja utrudnia przeniesienie informacji do końca.

**Koszt a jakość.** Czasy uczenia (2,1–2,7 s) są niemal identyczne we wszystkich konfiguracjach i nie rosną wyraźnie z L, więc przy tak małych modelach i jednym uruchomieniu nie da się na ich podstawie ustalić rankingu. Poprawa nie wiązała się z liczbą parametrów: jedyny model, który się nauczył (GRU, 13 121 parametrów), ma mniej parametrów niż Conv1D (25 217), a Conv1D nie poprawił wyniku mimo największej liczby parametrów. LSTM, który ma więcej parametrów niż GRU, nie nauczył się wcale.

**Stan ukryty SimpleRNN (64 -> 128).** Test MSE wzrósł z 0,1569 do 0,1713 (+0,0144, powyżej stałej 0,1572), przy 3,85-krotnie większej liczbie parametrów (4 417 -> 17 025) i podobnym czasie (0,81×; różnica w granicach niepewności pomiaru). Większy stan nie pomógł: zwiększa pojemność, ale nie zmienia mnożenia pochodnych w czasie, więc nie usuwa problemu zanikającego gradientu.

**Dylatacje Conv1D ([1,2,4] -> [1,1,1]).** Pole recepcyjne spadło z 15 do 7 pozycji, a liczba parametrów i czas pozostały bez zmian. Tylko pole 15 może w jednej cesze objąć znacznik i próbkę oddaloną o 8 kroków; przy polu 7 jest to niemożliwe. Test MSE obu wariantów jest jednak niemal identyczny (0,1542 vs 0,1543) i oba leżą na poziomie stałej (0,1572). Eksperyment nie potwierdza więc wpływu pola recepcyjnego: samo pole ≥ 9 nie wystarczyło, aby Conv1D nauczył się zadania w 30 epokach na 4000 przykładach.

**Przykład błędny.** Dla Conv1D (L = 100, przykład nr 192) prawidłowa suma 0,1089 jest bardzo mała (obie wskazane wartości bliskie 0), a predykcja wynosi 1,0635, czyli ok. średniej celu. Duży błąd (+0,9547) wynika więc z tego, że model zwraca wartość bliską stałej niezależnie od wejścia, a ten przykład leży daleko od średniej. Conv1D liczy wynik z mapy cech uśrednionej po całej sekwencji; uśrednianie rozmywa dwa istotne wkłady wśród 100 pozycji, a nauczenie się powiązania znacznika z próbką sprzed 8 kroków nie nastąpiło. Błąd nie zależy od położenia wskazanych próbek (31 i 63 kroki od końca w tym przykładzie): ρ Spearmana między |błędem| a odległością najwcześniej wskazanej próbki od końca wynosi −0,018, średni |błąd| to 0,28–0,35 we wszystkich przedziałach, a średnia ważona MSE ≈ 0,154. Potwierdza to, że model nie wykorzystuje znaczników.

<img width="689" height="390" alt="image" src="https://github.com/user-attachments/assets/3890ff7a-d0ac-443c-8635-36c84067894c" />


**Ograniczenia.** Jedno ziarno i jedna para zbiorów dla każdego L, więc nie oceniono stabilności (wynik GRU dla L = 50 trzeba by powtórzyć dla kilku ziaren); hiperparametry (30 epok, lr 5·10⁻³, 4000 próbek) nie były dostrajane, a brak nauki przy L = 100 może wynikać z zbyt krótkiego uczenia; modele różnią się pojemnością, więc porównanie nie izoluje wpływu samej architektury; czas zależy od urządzenia; ConvNet nie jest przyczynowy (symetryczny padding) i nie jest implementacją WaveNet.

## 5. Wnioski

1. Przy L = 50 tylko GRU pokonał stałą predykcję (test MSE 0,0893 wobec 0,1666); SimpleRNN, LSTM i Conv1D pozostały na jej poziomie. Przy L = 100 żaden model nie pokonał stałej predykcji (0,1542–0,1570 wobec 0,1572).
2. Wydłużenie sekwencji z 50 do 100 kroków zlikwidowało przewagę GRU (stosunek do stałej 0,536 -> 0,981); czas uczenia praktycznie się nie zmienił.
3. Zwiększenie stanu ukrytego SimpleRNN do 128 nie poprawiło wyniku (+0,0144) przy 3,85-krotnie większej liczbie parametrów.
4. Zmiana dylatacji Conv1D z [1,2,4] na [1,1,1] (pole 15 -> 7) nie zmieniła wyniku (0,1542 vs 0,1543), bo żadna wersja nie nauczyła się zadania; wpływu pola recepcyjnego nie udało się potwierdzić.
5. Poprawa nie była związana z liczbą parametrów (GRU: 13 121, Conv1D: 25 217), a pomiary czasu nie pozwalają na ranking architektur.
6. Kierunki poprawy: więcej epok i danych, dostrojenie szybkości uczenia na osobnym zbiorze walidacyjnym, powtórzenia z kilkoma ziarnami (ocena stabilności wyniku GRU), wstępne uczenie na krótszych sekwencjach, a dla Conv1D odczyt zachowujący informację o położeniu zamiast uśredniania po całym czasie.
