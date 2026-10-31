**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy

**Zajęcia nr / temat:** 3. Prognozowanie szeregów czasowych — prognoza wielokrokowa i porównanie z baseline'ami

**Autor / grupa:** Maja Świniarska GL.04

**Data:** 29.10.2026

## 1. Cel ćwiczenia

Sprawdzenie, czy GRU wnosi coś ponad proste reguły (naiwną, sezonową, średnią ruchomą) w prognozie jedno- i 14-dniowej
przy walidacji czasowej oraz ocena wpływu długości okna (3, 7, 28 dni).

## 2. Metoda i implementacja

- **Dane:** dzienna liczba odsłon hasła „Python" w polskiej Wikipedii (API Wikimedia), 1.01.2020–31.12.2022, 1096 dni,
  bez braków. Szereg ma rytm tygodniowy (wtorek ≈ 367, sobota ≈ 217 odsłon) i trend malejący (średnia roczna 343 → 316 → 282).
  Źródło: https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/pl.wikipedia/all-access/user/Python/daily/20200101/20221231 (kopia: python_wiki_pl.csv)
- **Podział chronologiczny:** trening 767 dni (do 5.02.2022), walidacja 164 (do 19.07.2022), test 165 (20.07–31.12.2022).
  Standaryzacja z μ ≈ 333,6 i σ ≈ 115,1 liczona wyłącznie z treningu; metryki w oryginalnej skali.
- **Model:** `nn.GRU(1, 32)` + `nn.Linear(32, 1)`, MSE, Adam (lr = 1e-2), 40 epok, batch 64, OKNO = 28, wagi z epoki
  o najniższej stracie walidacyjnej. Kod: `cwiczenie.ipynb`.
- **Ocena wielokrokowa:** prognoza rekurencyjna, H = 14, 11 startów (t0 = 931, 945, …, 1071). Baseline sezonowy powtarza
  ostatni znany tydzień, naiwny - ostatnią znaną wartość. Wszystkie metody oceniane na tych samych dniach.
- **Granica szumu (oszacowanie):** σ̂ z reszt po odjęciu średniej kroczącej (29 dni) i profilu dnia tygodnia: ≈ 57 (trening),
  ≈ 36 (test). Orientacyjnie MAE ≈ 0,8·σ̂, RMSE ≈ σ̂, czyli **na teście MAE ≈ 29, RMSE ≈ 36**.

## 3. Wyniki

### Krzywe uczenia (OKNO = 28, ziarno 0)

<img width="726" height="355" alt="image" src="https://github.com/user-attachments/assets/d265892b-0e00-4f46-878a-0f0ad9e64a70" />


Najlepsza epoka: **36 z 40** (strata walidacyjna 0,1892, treningowa 0,1637; ostatnia walidacyjna 0,1953).

### Test: prognoza 1-krokowa i 14-dniowa (GRU: OKNO 28, ziarno 0)

| metoda | MAE 1-krok | RMSE 1-krok | MAE 14-krok | RMSE 14-krok |
|---|---|---|---|---|
| GRU | **31,79** | **41,15** | 39,77 | 52,09 |
| sezonowy | 37,73 | 50,82 | **36,64** | **48,15** |
| naiwny | 45,15 | 58,26 | 50,13 | 65,78 |
| *śr. ruchoma (tylko 1-krok)* | *49,13* | *61,10* | — | — |
| *oszac. granica szumu* | *≈ 29* | *≈ 36* | — | — |

<img width="695" height="355" alt="image" src="https://github.com/user-attachments/assets/f77f5f90-bfa5-4914-b376-b43fbfc93f81" />

<img width="908" height="297" alt="image" src="https://github.com/user-attachments/assets/0825bacd-4e2b-4dd2-8c74-414f1393df8f" />



### Wpływ długości okna (2 ziarna)

| OKNO | ziarno | strata walid. | MAE 1-krok | MAE 14-krok |
|---|---|---|---|---|
| 3 | 0 / 1 | 0,3391 / 0,3451 | 46,52 / 45,13 | 72,35 / 55,08 |
| 7 | 0 / 1 | 0,2344 / 0,2274 | 32,71 / 32,94 | 33,91 / 33,85 |
| 28 | 0 / 1 | 0,1892 / 0,1854 | 31,79 / 32,20 | 39,77 / 32,79 |

Średnie: okno 3 — 0,342 / 45,8 / 63,7; okno 7 — 0,231 / 32,8 / 33,9; okno 28 — 0,187 / 32,0 / 36,3
(strata walid. / MAE 1-krok / MAE 14-krok). Najlepsze okno wg walidacji: **28**.

## 4. Analiza i interpretacja

**Krzywe uczenia.** Obie straty spadają do ok. 15. epoki, potem walidacyjna stoi na ok. 0,19–0,20, a treningowa nadal
powoli maleje. Rozbieżność jest mała, więc przeuczenia praktycznie nie ma; najlepsza epoka blisko końca oznacza lekkie
niedotrenowanie, ale zysk z dłuższego treningu byłby mały.

**GRU a baseline'y.** W prognozie 1-krokowej GRU wygrywa ze wszystkimi (MAE 31,79 wobec 37,73 sezonowego, −16 %).
W prognozie 14-dniowej bije naiwny (39,77 wobec 50,13), ale przy ziarnie 0 przegrywa z sezonowym (36,64). Przy ziarnie 1
ten sam wariant daje 32,79, więc dla okna 28 wynik zależy od ziarna (średnia 36,3 ≈ sezonowy). Różnica 1- i 14-krokowa
wynika z kumulacji błędu w strategii rekurencyjnej: od drugiego kroku wejście zawiera własne prognozy modelu. Baseline
sezonowy używa tylko znanego tygodnia, więc jego błąd nie rośnie z horyzontem. Naiwny ma skoki błędu dla h = 4 i 11
(soboty), bo ignoruje rytm tygodniowy.

**Granica szumu.** Wynik 1-krokowy (MAE 31,79; RMSE 41,15) jest o ok. 3 odsłony (≈ 11 %) i ok. 5 odsłon (≈ 15 %) powyżej
granicy MAE ≈ 29 i RMSE ≈ 36. Granica jest tylko orientacyjna: na walidacji GRU ma RMSE ≈ 50, mniej niż oszacowane
σ̂ ≈ 58, czyli estymator zawyża szum. Na teście dodatkowo poziom spada (średnia ≈ 227 wobec ≈ 334 na treningu).

**Długość okna.** Okno 3 dni nie obejmuje rytmu tygodniowego: MAE 1-krokowe (45,8) jest równe naiwnemu (45,15), a
14-dniowe (63,7) gorsze od naiwnego. Okno 28 daje niewielką poprawę względem 7 (strata walidacyjna −19 %, MAE 1-krok
32,0 wobec 32,8), ale w prognozie 14-dniowej okno 7 jest lepsze i stabilniejsze (33,9 ± 0,04 wobec 36,3 ± 4,9).

**Rozrzut ziaren.** Różnica 3 → 7 dni jest dużo większa niż rozrzut ziaren, więc jest wiarygodna. Różnica 7 → 28 w stracie
walidacyjnej i MAE 1-krok przekracza rozrzut, ale przy 2 ziarnach wniosek jest wstępny; w MAE 14-krok jest mniejsza niż
rozrzut okna 28 (4,9), więc nieistotna.

**Modele mowy.** (1) Okno musi obejmować skalę zjawiska — tu ≥ 7 dni, w mowie kontekst kilkudziesięciu ramek (głoska
≈ 50–150 ms). (2) Kumulacja błędu w prognozie rekurencyjnej to ten sam mechanizm co w dekodowaniu autoregresyjnym
(Tacotron, WaveRNN, seq2seq). (3) Normalizacja z treningu odpowiada CMVN, a podział chronologiczny — podziałowi
rozłącznemu po mówcach. (4) Zawsze trzeba porównywać z prostym baseline'em.

## 5. Wnioski

1. GRU wyraźnie bije baseline'y w prognozie 1-krokowej (MAE 31,79 wobec 37,73 sezonowego) i jest blisko oszacowanej
   granicy szumu (MAE ≈ 29).
2. W prognozie 14-dniowej przewaga znika przez kumulację błędu: GRU bije naiwny, ale z sezonowym jest na równi lub
   gorszy, zależnie od ziarna.
3. Okno 3 dni jest za krótkie; według walidacji najlepsze jest 28 dni, choć zysk względem 7 dni jest mały, a w prognozie
   14-dniowej okno 7 jest stabilniejsze.
4. Ograniczenia: 11 startów, 2 ziarna, przesunięcie poziomu szeregu w teście, przybliżona granica szumu.
   Możliwe poprawy: strategia *direct* lub model z 14 wyjściami, cechy kalendarzowe, log-skala.

---



