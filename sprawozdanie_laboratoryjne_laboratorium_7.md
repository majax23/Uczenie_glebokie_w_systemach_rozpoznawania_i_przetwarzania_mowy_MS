# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy

**Zajęcia nr / temat:** 7. Projekt generacyjny, cz. 2 – budowa i uczenie Char-RNN

**Autor / grupa:** Maja Świniarska GL.04

**Data:** 26.11.2026

## 1. Cel ćwiczenia

Celem było zbudowanie modelu znakowego Char-RNN (Embedding → GRU/LSTM → warstwa liniowa), porównanie wariantów architektury w kontrolowanym pilotażu na danych z laboratorium 6 (*Quo vadis* Henryka Sienkiewicza), przeprowadzenie dłuższego treningu najlepszego wariantu oraz zapisanie checkpointu `best` wraz z konfiguracją i słownikiem do generowania tekstu na laboratorium 8.

## 2. Metoda i implementacja

### Narzędzia i środowisko

Python, PyTorch 2.14.1+cu130 (`nn.Embedding`, `nn.GRU`, `nn.LSTM`, `nn.Linear`, `CrossEntropyLoss`, `LambdaLR`), NumPy, pandas, matplotlib; środowisko Google Colab z akceleratorem **NVIDIA Tesla T4 (`cuda`)**. Ustalono ziarno losowania `SEED = 0`.

### Dane wejściowe i wybór `SEKW`

Wykorzystano ciągłe fragmenty korpusu oraz słownik z laboratorium 6 (`V = 83`, w tym `UNK = ''` pod indeksem 82). W zbiorach treningowym i walidacyjnym nie wystąpiły znaki spoza słownika, natomiast w teście odnotowano 3 wystąpienia `UNK` (wielka litera `Q`).

**Uzasadnienie wyboru `SEKW = 40`.** Okno o długości 40 znaków odpowiada średnio 6–7 wyrazom w tekście, co zapewnia wystarczający kontekst lokalny (fleksja, składnia krótkiej frazy), jednocześnie utrzymując niski koszt obliczeniowy aktualizacji i stabilną propagację gradientu w sieciach rekurencyjnych (w porównaniu do długich sekwencji typu `SEKW = 160`).

**Tabela 1.** Liczności zbiorów i batchy (`SEKW = 40`, `KROK = 1`)

| zbiór | znaków | par | rozmiar batcha | batchy | ocenianych pozycji |
|---|---:|---:|---:|---:|---:|
| trening | 869 557 | 869 517 | 64 (`drop_last=True`) | 13 586 | 34 780 160 |
| walidacja | 109 603 | 109 563 | 1024 (`drop_last=False`) | 107 | 4 382 520 |
| test | 107 421 | 107 381 | 1024 (`drop_last=False`) | 105 | 4 295 240 |

### Architektura, trening i walidacja

- **Model `CharRNN`:** warstwa zanurzeń `nn.Embedding(83, 64)` → warstwa rekurencyjna `GRU` lub `LSTM` (`hidden=256`, `batch_first=True`, `dropout=0.2` dla `layers > 1`) → warstwa liniowa `nn.Linear(256, 83)`. Każde okno w treningu rozpoczyna się od zerowego stanu ukrytego (`h=None`), a kolejne kroki wykorzystują *teacher forcing*.
- **Funkcja celu i ewaluacja:** `nn.CrossEntropyLoss` obliczana na spłaszczonych logitach `[B*T, V]`. Funkcja `ocen()` działa w trybie `model.eval()` i `torch.no_grad()`, ważąc stratę każdego batcha liczbą jego celów (`yb.numel()`), co eliminuje wpływ krótszego ostatniego batcha. Perplexity wyliczana jest jako $\text{PPL} = \exp(\text{średnia strata})$.
- **Kontrole poprawności:** przed treningiem model uzyskał stratę walidacyjną $4{,}431$ ($\text{PPL} = 84{,}0$, blisko wartości teoretycznej $\ln(83) \approx 4{,}419$ dla rozkładu jednostajnego). Testy jednostkowe potwierdziły niezależność `ocen()` od rozmiaru batcha ($4{,}430378$ dla `batch=1024` oraz `batch=777`), brak modyfikacji wag podczas ewaluacji oraz spadek straty na pojedynczym mini-batchu z $4{,}421$ do $0{,}037$ w 150 krokach.

**Tabela 2.** Konfiguracja treningu i pilotażu

| parametr | wartość |
|---|---|
| długość okna `SEKW` / krok | 40 / 1 |
| rozmiar słownika `V` | 83 |
| wymiar embeddingu `EMB` / stan ukryty `HIDDEN` | 64 / 256 |
| `dropout` | 0,2 (aktywny dla `layers >= 2`) |
| `BATCH` (trening) / `BATCH_EVAL` (walidacja, test) | 64 / 1024 |
| optymalizator / `LR` startowy | Adam / $10^{-3}$ |
| gradient clipping (`CLIP`) | 1,0 (norma $L_2$) |
| harmonogram LR (długi trening) | liniowy warmup 500 kroków + cosine decay do $0{,}05 \cdot \text{LR}$ ($5 \cdot 10^{-5}$) |
| walidacja / early stopping | co 2000 kroków oraz na koniec epoki / cierpliwość 8 walidacji |
| urządzenie / ziarno losowania | `cuda` (Tesla T4) / `SEED = 0` |

## 3. Wyniki

### Pilotaż — porównanie wariantów architektury

Porównano trzy warianty: `GRU-1` (1 warstwa GRU), `LSTM-1` (1 warstwa LSTM) oraz `GRU-2` (2 warstwy GRU). Na podstawie pomiaru czasu najwolniejszego wariantu (`GRU-2`: $6{,}16\text{ ms/krok}$) dla budżetu 60 s wyznaczono wspólny limit **8800 aktualizacji wag** ($0{,}65$ epoki, $22\,528\,000$ ocenianych pozycji na model) oraz 4 wspólne momenty walidacji: kroki `2200`, `4400`, `6600` i `8800`. Osobne generatory `torch.Generator(SEED=0)` oraz suma kontrolna batchy potwierdziły, że wszystkie warianty otrzymały identyczne batche w tej samej kolejności.

**Tabela 3.** Wyniki pilotażu (walidacyjna PPL, czas treningu, szybkość i liczba parametrów)

| metryka | GRU-1 | LSTM-1 | GRU-2 |
|---|---:|---:|---:|
| parametry | 273 939 | 356 371 | 668 691 |
| PPL val @ 2200 | 4,896 | 5,393 | **4,646** |
| PPL val @ 4400 | 4,633 | 4,819 | **4,416** |
| PPL val @ 6600 | 4,581 | 4,640 | **4,352** |
| PPL val @ 8800 | 4,558 | 4,556 | **4,315** |
| czas @ 2200 [s] | 10,8 | 11,3 | 14,7 |
| czas @ 4400 [s] | 21,2 | 22,1 | 29,8 |
| czas @ 6600 [s] | 31,8 | 33,1 | 44,4 |
| czas @ 8800 [s] | 42,2 | 44,1 | 59,1 |
| średni czas kroku [ms] | 4,79 | 5,01 | 6,72 |

<img width="790" height="340" alt="image" src="https://github.com/user-attachments/assets/2016c7db-06ec-47d3-ae6d-4e0b0a4ace61" />


*Rys. 1. Przebieg walidacyjnej perplexity w pilotażu dla trzech wariantów architektury (do kroku 8800).*

**Uzasadnienie wyboru wariantu.** Po 8800 krokach najniższą walidacyjną PPL uzyskał wariant **`GRU-2` ($\text{PPL} = 4{,}315$)**, wyprzedzając drugi model (`LSTM-1`, $\text{PPL} = 4{,}556$) o **5,59%** (powyżej progu podobieństwa 2%). Mimo większej liczby parametrów ($668\,691$) i dłuższego czasu kroku ($6{,}72\text{ ms}$), przewaga jakościowa `GRU-2` była wyraźna już od pierwszego punktu pomiarowego, dlatego ten wariant wybrano do dłuższego treningu.

### Dłuższy trening wybranego wariantu (`GRU-2`)

Budżet maksymalny ustalono na 10 epok ($135\,860$ kroków). Trening trwał **5,3 min** i został zatrzymany automatycznie przez *early stopping* w **kroku 38 000** (epoka 2,80) po 8 kolejnych walidacjach bez poprawy najlepszego wyniku. Najlepszy model (`char_rnn_best.pt`) zapisano w **kroku 24 000** (epoka 1,77).

**Tabela 4.** Wybrane punkty kontrolne z przebiegu treningu `GRU-2`

| krok | epoka | learning rate | strata treningowa | PPL treningowa | strata walidacyjna | PPL walidacyjna | uwaga |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 000 | 0,15 | $1{,}00 \cdot 10^{-3}$ | 1,9214 | 6,8303 | 1,5583 | 4,7505 | |
| 6 000 | 0,44 | $9{,}96 \cdot 10^{-4}$ | 1,4253 | 4,1592 | 1,4755 | 4,3731 | |
| 13 586 | 1,00 | $9{,}78 \cdot 10^{-4}$ | 1,3485 | 3,8517 | 1,4606 | 4,3085 | koniec epoki 1 |
| 18 000 | 1,32 | $9{,}61 \cdot 10^{-4}$ | 1,3303 | 3,7821 | 1,4584 | 4,2992 | |
| **24 000** | **1,77** | **$9{,}31 \cdot 10^{-4}$** | **1,3136** | **3,7197** | **1,4577** | **4,2961** | **checkpoint `best`** |
| 27 172 | 2,00 | $9{,}12 \cdot 10^{-4}$ | 1,3054 | 3,6892 | 1,4580 | 4,2973 | koniec epoki 2 |
| 32 000 | 2,36 | $8{,}79 \cdot 10^{-4}$ | 1,2953 | 3,6520 | 1,4595 | 4,3039 | |
| 38 000 | 2,80 | $8{,}31 \cdot 10^{-4}$ | 1,2867 | 3,6210 | 1,4592 | 4,3023 | early stopping |

<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/07ac9b21-aca4-47dd-b47a-d0588bc3d9fc" />


*Rys. 2. Strata treningowa i walidacyjna (cross-entropy) oraz harmonogram współczynnika uczenia podczas treningu `GRU-2`.*

### Ocena końcowa na zbiorze testowym

Po wczytaniu zapisanego checkpointu `char_rnn_best.pt` (krok 24 000) obliczono metryki na odłożonym zbiorze testowym ($4\,295\,240$ ocenianych pozycji).

**Tabela 5.** Wyniki checkpointu `best` (`GRU-2`, krok 24 000) na walidacji i teście

| zbiór | ocenianych pozycji | strata (cross-entropy) | perplexity (PPL) |
|---|---:|---:|---:|
| walidacja | 4 382 520 | 1,4577 | 4,2961 |
| test | 4 295 240 | 1,4709 | 4,3531 |

### Sprawdzenie checkpointu w świeżej sesji

W osobnej komórce zdefiniowano klasę `CharRNN`, wczytano plik `char_rnn.pt` (kopia `char_rnn_best.pt`), odtworzono architekturę wyłącznie na podstawie zapisanego słownika `config` (`{'emb': 64, 'hidden': 256, 'typ': 'GRU', 'layers': 2, 'dropout': 0.2}`) oraz mapowań `stoi`/`itos`, a następnie wygenerowano kontynuacje o długości 300 znaków (temperatura $T = 0{,}8$, `torch.manual_seed(0)`):

- **Początek:** `'Petroniusz '`
  > podniósł się i zaprosiła się śmiercią, ale nie mogę odpowiedzieć przytomność, którą naraz jeszcze odnaleźli w górę, czekać w więzieniu, na linie, co czynił je do Petroniusza i z niewolnikami, ale na miastem godnością przez słowa mroki.
  >
  > — Gdyby on, przy niej nadziej, i Ligia miała mi posłuchu ligijs

- **Początek:** `'Winicjusz rzekł: '`
  > „Idziemy do piersi i włosów i prowadził wypadkiem z tego posilich młodego domu, z zasługa o spokoju w razie szli nagle jako prostoty u Eurycjusza, wydał się i czy wiesz, co chwila jest człowiek i jego miłość arabskich i męką, i wiatr po chwili wyznawały mu zwiększyli z przepierać jej wśród tych rzęś

### Zadanie dodatkowe: wpływ długości okna `SEKW` przy stałej liczbie kroków

Przeprowadzono dwa treningi wybranej architektury `GRU-2` dla `SEKW = 20` oraz `SEKW = 80` przy identycznej liczbie aktualizacji wag (`8800` kroków, `BATCH = 64`, `LR = 1e-3`). Oba modele oceniono na tym samym zbiorze walidacyjnym o stałej długości okna `SEKW_OCENA = 80`.

**Tabela 6.** Porównanie treningu `GRU-2` dla `SEKW = 20` i `SEKW = 80` przy 8800 aktualizacjach

| SEKW (trening) | aktualizacji wag | pozycji w treningu | czas treningu [s] | PPL val (`SEKW=80`) |
|---:|---:|---:|---:|---:|
| 20 | 8 800 | 11 264 000 | 45,8 | 4,300 |
| 80 | 8 800 | 45 056 000 | 104,1 | 4,151 |

Zwiększenie `SEKW` z 20 do 80 obniżyło walidacyjną PPL o **3,5%** (z $4{,}300$ do $4{,}151$), ale wydłużyło czas treningu o **127,3%** (z $45{,}8\text{ s}$ do $104{,}1\text{ s}$) i zwiększyło liczbę przetworzonych pozycji w treningu **4-krotnie** (z $11{,}26\text{ mln}$ do $45{,}06\text{ mln}$).

## 4. Analiza i interpretacja

1. **Porównanie architektur (pilotaż).** W wariantach jednowarstwowych `GRU-1` uczy się na początku szybciej od `LSTM-1` (PPL $4{,}896$ vs $5{,}393$ w kroku 2200) ze względu na mniejszą liczbę bramek i parametrów ($273{,}9\text{ tys.}$ vs $356{,}4\text{ tys.}$), ale pod koniec pilotażu oba modele wyrównują wyniki ($\text{PPL} \approx 4{,}56$). Dwuwarstwowy `GRU-2` ma ponad dwukrotnie większą pojemność ($668{,}7\text{ tys.}$ parametrów), co pozwala lepiej uchwycić zależności wieloznakowe i obniżyć PPL do $4{,}315$ kosztem wolniejszego kroku ($+40\%$ czasu względem `GRU-1`).
2. **Przebieg uczenia i brak przeuczenia.** Najszybszy spadek straty następuje w pierwszej połowie pierwszej epoki (do kroku 8000 strata walidacyjna spada z $4{,}431$ do $1{,}468$). Po osiągnięciu minimum w kroku 24 000 ($\text{strata val} = 1{,}4577$, $\text{PPL} = 4{,}296$) strata treningowa nadal powoli maleje (do $1{,}2867$), natomiast strata walidacyjna wchodzi w stabilne plateau ($1{,}458\text{–}1{,}462$, wzrost zaledwie o $+0{,}0015$ do końca treningu). Oznacza to brak wyraźnego przeuczenia i uzasadnia zadziałanie *early stopping*. Warto zaznaczyć, że strata treningowa jest raportowana przy włączonym `dropout=0.2`, dlatego ma wartości nieco niższe dopiero po pierwszym tysiącu kroków.
3. **Generalizacja na zbiór testowy.** Strata na teście ($1{,}4709$, $\text{PPL} = 4{,}3531$) jest bardzo zbliżona do wyniku walidacyjnego ($\Delta\text{PPL} = +0{,}057$), co potwierdza dobrą generalizację modelu na końcowe rozdziały powieści (mimo 3 wystąpień znaku `UNK`).
4. **Jakość próbek tekstowych.** Wygenerowany tekst zachowuje poprawną polską ortografię, interpunkcję dialogową (myślniki, cudzysłowy), fleksję oraz słownictwo i imiona bohaterów *Quo vadis* (*Petroniusz*, *Ligia*, *Eurycjusz*). Widoczne są natomiast ograniczenia małego modelu znakowego o krótkiej pamięci: brak spójności semantycznej na przestrzeni kilku zdań oraz sporadyczne neologizmy (*posilich*).
5. **Interpretacja eksperymentu z `SEKW`.** Porównanie `SEKW = 20` i `SEKW = 80` przy stałej liczbie kroków nie jest porównaniem przy tym samym koszcie obliczeniowym ani przy tej samej ilości danych — okno 4-krotnie dłuższe dostarcza w każdym kroku 4 razy więcej celów do obliczenia gradientu. Niższa PPL dla `SEKW = 80` wynika więc zarówno z szerszego kontekstu (do 80 znaków przy ocenie), jak i z 4-krotnie większej liczby przetworzonych tokenów podczas treningu.

## 5. Wnioski

1. W pilotażu na korpusie *Quo vadis* dwuwarstwowa sieć `GRU-2` osiągnęła wyraźnie lepszą perplexity ($\text{PPL} = 4{,}315$) niż warianty jednowarstwowe `GRU-1` ($4{,}558$) i `LSTM-1` ($4{,}556$), uzasadniając wyższy koszt obliczeniowy.
2. Harmonogram *warmup + cosine decay* oraz *early stopping* pozwoliły osiągnąć minimum straty walidacyjnej w kroku 24 000 ($\text{PPL} = 4{,}2961$) i zatrzymać trening w fazie plateau bez przeuczenia modelu.
3. Model dobrze generalizuje na odłożony zbiór testowy ($\text{PPL} = 4{,}3531$), a jego poprawny zapis i odtworzenie w świeżej sesji potwierdziły gotowość checkpointu `char_rnn.pt` (`char_rnn_best.pt`) do dalszych eksperymentów z generowaniem tekstu na laboratorium 8.

