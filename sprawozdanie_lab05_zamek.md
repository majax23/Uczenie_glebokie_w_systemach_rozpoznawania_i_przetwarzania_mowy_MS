# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 5 - Wprowadzenie do NLP (BPE, Word2Vec, mechanizm uwagi)
**Autor / grupa:** ✏️ **[UZUPEŁNIJ]**
**Data:** ✏️ **[UZUPEŁNIJ]**

> **Do uzupełnienia przed oddaniem (usuń ten blok):** ✏️ imię, grupa, data; ✏️ link do repozytorium (sekcja 2.1); 📋 trzy wykresy z plików PNG zapisanych przez notatnik: `bpe_tokeny_na_slowo.png` (sekcja 3.1), `pca_zamek.png` (3.3), `uwaga_zamek.png` (3.4). Wszystkie tabele i liczby są już wpisane z notatnika.

## 1. Cel ćwiczenia

Na własnym korpusie 60 polskich zdań zbadano: (1) wpływ rozmiaru słownika BPE na podział form fleksyjnych widzianych i niewidzianych w treningu, (2) wpływ proporcji kontekstów wyrazu wieloznacznego „zamek” (budowla / mechanizm) na jego sąsiadów w Word2Vec, (3) czy w embeddingach widać grupy tematyczne (PCA, cosinus). Dodatkowo (4) pokazano na tych samych wektorach mechanizm uwagi.

## 2. Metoda i implementacja

### 2.1 Narzędzia

`tokenizers` 0.23.3 (BPE, normalizer `Lowercase`, pretokenizer `Whitespace`), `gensim` 4.4.0 (Word2Vec, Skip-gram), `scikit-learn` (PCA), `numpy`, `pandas`, `matplotlib`; Google Colab. Notebook: ✏️ **[UZUPEŁNIJ: link do repozytorium / `LAB5MOWA.ipynb`]**

### 2.2 Korpus i próbka testowa

**Źródło:** zdania z puli A i B napisane na potrzeby ćwiczenia (częściowo z pomocą AI); zdania z puli C zaczerpnięte z notatnika ćwiczeniowego. Pełna lista: Załącznik A.

| Pula | Opis | Zdań | Tokenów | „zamek” |
|---|---|---|---|---|
| A | zamek = budowla (król, rycerz, wieża, mury, fosa) | 20 | 222 | 20 |
| B | zamek = mechanizm (klucz, drzwi, ślusarz, wytrych, klamka) | 20 | 200 | 20 |
| C | bez „zamek” (ML, zwierzęta, język polski) | 20 | 149 | 0 |
| razem | | 60 | 571 | 40 |

Różnych form wyrazowych: 364. W każdym zdaniu A i B „zamek” występuje raz; innych form tego wyrazu w korpusie nie ma (kontrola `assert`). Słowa tematyczne występują w dokładnie takiej formie, jak na listach grup (bez lematyzacji).

**Przygotowanie tekstu:** Word2Vec - małe litery, `re.findall(r'\w+', ...)` (bez interpunkcji), bez lematyzacji i bez usuwania słów funkcyjnych. BPE - `Lowercase` + `Whitespace`, trening na wszystkich 60 zdaniach.

**Próbka BPE** - 8 par (forma widziana → niewidziana, liczba wystąpień formy widzianej w korpusie):

| widziana (wyst.) | niewidziana (0 wyst.) |
|---|---|
| zamek (40) | zamkami |
| król (5) | królami |
| klucz (7) | kluczami |
| drzwi (5) | drzwiami |
| fosa (3) | fosami |
| wieża (4) | wieżami |
| rycerz (5) | rycerzami |
| klamka (3) | klamkami |

### 2.3 BPE

**Hipoteza (przed uruchomieniem):**
> Zwiększanie limitu słownika dodaje kolejne scalenia wyuczone na korpusie. Formy widziane w treningu będą więc coraz częściej scalane do jednego tokenu (średnia liczba tokenów na słowo spadnie w stronę 1). Formy niewidziane spadną wolniej: ich rdzeń zostanie scalony tak jak w formie widzianej, ale końcówka -ami będzie składana z krótkich fragmentów. Token `[UNK]` nie powinien się pojawić, bo wszystkie litery tych form występują w korpusie treningowym (alfabet BPE jest zbudowany ze znaków z korpusu).

**Konfiguracja:** trzy tokenizery na całym korpusie, zmieniano tylko `vocab_size` (`min_frequency=1`, `special_tokens=['[UNK]']`); każdą z 16 form kodowano osobno.

| Ustawiony limit | Faktyczny rozmiar \|V\| |
|---|---|
| 100 | 100 |
| 250 | 250 |
| 500 | 500 |

Limity wybrano na podstawie przeglądu pomocniczego (limity 60-700 co 40; w tabeli wyników nie uwzględniony).

```python
def train_bpe(limit):
    tok = Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=limit, special_tokens=['[UNK]'],
                                  min_frequency=1, show_progress=False)
    tok.train_from_iterator(KORPUS, trainer)
    return tok
```

### 2.4 Word2Vec

Dwa warianty po 36 zdań i 16 wystąpień „zamek”, ta sama pula C (20 zdań): **V1** = C + 12 A + 4 B, **V2** = C + 4 A + 12 B. Zdania tasowane tym samym ziarnem. Identyczne ustawienia obu modeli:

`vector_size=50, window=4, min_count=2, sg=1, epochs=200, negative=5, sample=0, seed=0, workers=1, hashfxn=crc32` (domyślny `hash()` zależy od `PYTHONHASHSEED`; `sample=0`, bo korpus jest mały). Podobieństwo: `cos(u, v) = (u · v) / (‖u‖ ‖v‖)`.

### 2.5 Grupy tematyczne i PCA

Grupy wybrane przed treningiem:

| Grupa | Słowa | Uzasadnienie |
|---|---|---|
| 1 - budowla | rycerz, król, wieża, mury, fosa | osoby i elementy architektury zamku-twierdzy |
| 2 - mechanizm | klucz, drzwi, ślusarz, wytrych, klamka | elementy, narzędzia i osoba związane z zamkiem w drzwiach |

W V1 każde z 10 słów ma ≥ `min_count` = 2 wystąpienia (od 2 do 5; `assert`). Jeden rzut PCA (2 składowe) wektorów M1: 10 słów + „zamek”. Średni cosinus na pełnych 50-wymiarowych wektorach M1, tylko słowa z grup, każda para raz: 10 par wewnątrz G1, 10 wewnątrz G2, 25 między grupami.

### 2.6 Mechanizm uwagi (część dodatkowa)

Uproszczona uwaga jednogłowicowa bez uczonych projekcji (`V = K`): `q` = wektor „zamek”, `K` = wektory słów zdania; `s_i = (q·k_i)/√50`, `α = softmax(s)`, `c = α @ K` (50 współrzędnych). Dwa zdania: A12 „Rycerz i król ucztowali w wielkiej sali, a zamek rozbrzmiewał muzyką.” i B1 „Ślusarz wymienił zamek, bo klucz nie pasował do drzwi, a klamka luźno wisiała.”; dwa warianty kluczy: wszystkie słowa zdania albo tylko otoczenie. Osobny model z `min_count=1` (każde słowo zdania ma wektor).

## 3. Wyniki

### 3.1 BPE

| Limit | \|V\| | Grupa form | Śr. tokenów / słowo | Formy z `[UNK]` (z 8) |
|---|---|---|---|
| 100 | 100 | widziane | 2,75 | 0 |
| 100 | 100 | niewidziane | 4,50 | 0 |
| 250 | 250 | widziane | 1,00 | 0 |
| 250 | 250 | niewidziane | 2,88 | 0 |
| 500 | 500 | widziane | 1,00 | 0 |
| 500 | 500 | niewidziane | 2,88 | 0 |

Baseline = najmniejszy słownik (\|V\|=100): przy \|V\|=500 widziane −1,75 tokenu/słowo (−64%), niewidziane −1,62 (−36%). Przegląd pomocniczy: formy widziane osiągają 1,00 od \|V\|=220; niewidziane stoją na 2,88 w zakresie 220-540 i spadają do 2,62 dopiero od 580.

> 📋 **[WKLEJ WYKRES]** `bpe_tokeny_na_slowo.png` (śr. tokenów/słowo vs \|V\|, widziane i niewidziane).

Podziały wszystkich form:

| Forma | \|V\|=100 | \|V\|=250 | \|V\|=500 |
|---|---|---|---|
| **zamek** (w.) | zamek | zamek | zamek |
| **król** (w.) | k \| ró \| l | król | król |
| **klucz** (w.) | klucz | klucz | klucz |
| **drzwi** (w.) | d \| rz \| wi | drzwi | drzwi |
| **fosa** (w.) | f \| o \| sa | fosa | fosa |
| **wieża** (w.) | wie \| ż \| a | wieża | wieża |
| **rycerz** (w.) | ry \| ce \| rz | rycerz | rycerz |
| **klamka** (w.) | k \| l \| a \| m \| ka | klamka | klamka |
| *zamkami* (n.) | zam \| ka \| mi | zam \| ka \| mi | zam \| ka \| mi |
| *królami* (n.) | k \| ró \| l \| a \| mi | kró \| la \| mi | kró \| la \| mi |
| *kluczami* (n.) | klu \| c \| zam \| i | klu \| c \| zam \| i | klu \| c \| zam \| i |
| *drzwiami* (n.) | d \| rz \| wi \| a \| mi | drzwi \| a \| mi | drzwi \| a \| mi |
| *fosami* (n.) | f \| o \| sa \| mi | fosa \| mi | fosa \| mi |
| *wieżami* (n.) | wie \| ż \| a \| mi | wieża \| mi | wieża \| mi |
| *rycerzami* (n.) | ry \| ce \| r \| zam \| i | ryce \| r \| zam \| i | ryce \| r \| zam \| i |
| *klamkami* (n.) | k \| l \| a \| m \| ka \| mi | klamka \| mi | klamka \| mi |

(w. = widziana, n. = niewidziana)

**Dwie niewidziane formy przy trzech limitach:**

| Forma | \|V\|=100 → 250 | zniknęły | pojawiły się | fragment w formach treningowych | \|V\|=250 → 500 |
|---|---|---|---|---|---|
| królami | k \| ró \| l \| a \| mi → kró \| la \| mi | k, ró, l, a | kró, la | `kró`: król (×5); `la`: klamka (×3), kasztelan, klatki, latem, oddzielała (po ×1) | bez zmian |
| drzwiami | d \| rz \| wi \| a \| mi → drzwi \| a \| mi | d, rz, wi | drzwi | `drzwi`: drzwi (×5) | bez zmian |

**Sprawdzenie hipotezy** (|V| 100 → 500): (a) widziane 2,75 → 1,00 (końcowa ≤ 1,1: tak); (b) niewidziane 4,50 → 2,88, spadek mniejszy niż u widzianych: tak; (c) niewidziane po największym limicie nadal > 1 token: tak (2,88); (d) `[UNK]`: 0 form.

Czy wspólny początek formy widzianej i niewidzianej jest w niewidzianej jednym tokenem (|V|=500):

| Forma niewidziana | Wspólny początek | Podział | Jeden token |
|---|---|---|---|
| zamkami | zam | zam \| ka \| mi | tak (artefakt, patrz 4.1) |
| królami | król | kró \| la \| mi | nie |
| kluczami | klucz | klu \| c \| zam \| i | nie |
| drzwiami | drzwi | drzwi \| a \| mi | tak |
| fosami | fosa | fosa \| mi | tak |
| wieżami | wieża | wieża \| mi | tak |
| rycerzami | rycerz | ryce \| r \| zam \| i | nie |
| klamkami | klamka | klamka \| mi | tak |

Rangi reguł scalenia (mniejsza = wcześniej wyuczona; |V|=500): `zam` 3, `mi` 32, `la` 66, `klu` 60, `kró` 89, `drzwi` 99.

### 3.2 Word2Vec

Słownik modeli: M1 42 słowa, M2 40; w obu wariantach 16/16 zdań ze „zamek” zawiera słowo z grup.

**Pięciu najbliższych sąsiadów „zamek” (cosinus):**

| Miejsce | M1 (V1: 12 A + 4 B) | cos | M2 (V2: 4 A + 12 B) | cos |
|---|---|---|---|---|
| 1 | klucz | 0,639 | do | 0,564 |
| 2 | więc | 0,583 | nie | 0,560 |
| 3 | chroniła | 0,548 | z | 0,510 |
| 4 | wieża | 0,514 | wytrych | 0,492 |
| 5 | wytrych | 0,490 | klamka | 0,484 |

Bez słów funkcyjnych: M1 - klucz 0,639, chroniła 0,548, wieża 0,514, wytrych 0,490, fosa 0,467; M2 - wytrych 0,492, klamka 0,484, drzwi 0,483, klucz 0,471, były 0,463.

**Zmiany:** weszły do top-5 w M2: do, nie, z, klamka; wypadły: klucz, więc, chroniła, wieża; wspólne: wytrych.

| Słowo | Częstość V1 / V2 | Zdań ze „zamek” V1 / V2 | Wynik w M2 |
|---|---|---|---|
| klucz | 3 / 6 | 3 / 6 | cos 0,471, miejsce 8 |
| więc | 3 / 4 | 3 / 4 | cos 0,479, miejsce 7 |
| chroniła | 2 / 1 | 2 / 1 | brak wektora (1 < `min_count`=2) |
| wieża | 4 / 2 | 4 / 2 | cos 0,320, miejsce 19 |
| klamka | 2 / 3 | 2 / 3 | weszła do top-5 |
| nie | 1 / 3 | 1 / 3 | weszło do top-5 |

**Odporność na ziarno** (10 ziaren, średnie cos „zamek” do słów grupy):

| Wariant | cos → G1 (budowla) | cos → G2 (mechanizm) | G1 − G2 |
|---|---|---|---|
| V1 (12 A + 4 B) | 0,475 ± 0,007 | 0,440 ± 0,009 | +0,035 ± 0,014 |
| V2 (4 A + 12 B) | 0,357 ± 0,012 | 0,440 ± 0,010 | −0,083 ± 0,012 |

Oczekiwany znak różnicy (V1 > 0, V2 < 0): 10/10 ziaren w obu wariantach.

### 3.3 Grupy tematyczne

PCA (M1): PC1 = 35,9%, PC2 = 19,8%, razem 55,7% wariancji.

> 📋 **[WKLEJ WYKRES]** `pca_zamek.png` (rzut PCA 2D, grupy w kolorach, „zamek” wyróżniony gwiazdką).

| Zestaw par | Liczba par | Średni cosinus |
|---|---|---|
| wewnątrz grupy 1 | 10 | 0,615 |
| wewnątrz grupy 2 | 10 | 0,703 |
| między grupami | 25 | 0,355 |

Różnice: G1 − między = +0,260, G2 − między = +0,348.

| Para | cos | Wspólne zdania w V1 | Częstości w V1 |
|---|---|---|---|
| mury - fosa (najwyższa wewnątrz grup) | 0,852 | 2 | 3 i 3 |
| rycerz - wieża (najniższa wewnątrz grup) | 0,473 | 0 | 5 i 4 |
| mury - klamka (najwyższa między grupami) | 0,576 | 0 | 3 i 2 |

### 3.4 Mechanizm uwagi (część dodatkowa)

Największe wagi α: wariant ze „zamek” w kluczach - „zamek” 0,396 (zdanie A12) i 0,337 (B1); wariant „tylko otoczenie” - A12: król 0,132, ucztowali 0,117, sali 0,115 (najmniejsza: w 0,059); B1: bo 0,105, klucz 0,104, drzwi 0,096 (najmniejsza: wisiała 0,059).

> 📋 **[WKLEJ WYKRES]** `uwaga_zamek.png` (wagi α, siatka 2×2).

| Zdanie | Klucze | cos(c, e_zamek) | cos(c, centroid G1) | cos(c, centroid G2) |
|---|---|---|---|---|
| A12 (budowla) | ze „zamek” | 0,874 | 0,733 | 0,611 |
| A12 (budowla) | tylko otoczenie | 0,500 | 0,770 | 0,576 |
| B1 (mechanizm) | ze „zamek” | 0,820 | 0,619 | 0,874 |
| B1 (mechanizm) | tylko otoczenie | 0,492 | 0,559 | 0,964 |

cos(c_budowla, c_mechanizm) = 0,850 (ze „zamek”) i 0,607 (tylko otoczenie).

## 4. Analiza i interpretacja

### 4.1 BPE

- **Formy widziane** (hipoteza potwierdzona): 2,75 → 1,00 tokenu/słowo; od \|V\|=220 wszystkie 8 form to pojedyncze tokeny, więc \|V\|=250 i 500 dają identyczny wynik.
- **Formy niewidziane** (potwierdzona): spadek 4,50 → 2,88 jest względnie mniejszy (−36% wobec −64%) i zatrzymuje się powyżej 1, bo pełne formy z -ami nie występują w korpusie i nie mają wpisu w słowniku. `[UNK]` = 0, bo wszystkie litery tych form są w korpusie.
- **Twierdzenie o rdzeniu** (potwierdzone tylko częściowo): rdzeń jest jednym tokenem w 4 z 8 form (drzwiami, fosami, wieżami, klamkami). Wynik „tak” dla zamkami jest artefaktem testu: wspólny początek „zamek”/„zamkami” to „zam”, a prawdziwy rdzeń „zamk” jest rozbity (zam \| ka \| mi). Rdzeń nie jest scalony w królami, kluczami, rycerzami.
- **Przyczyna to kolejność reguł BPE**, nie to, że pełny rdzeń jest w słowniku: reguła tworząca `zam` ma rangę 3 (słowo „zamek” × 40), więc zabiera „z” z rdzeni klucz- i rycerz- (klu \| c \| **zam** \| i, ryce \| r \| **zam** \| i), mimo że `klucz` i `rycerz` są pojedynczymi tokenami w formach widzianych. W królami reguła `la` (66) jest wcześniejsza niż `kró` (89) i zabiera „l”, więc `król` nie powstaje.
- **Zmiany podziałów (2.5):** przy 100 → 250 w królami zniknęły k, ró, l, a, a pojawiły się `kró` (w treningowym „król” ×5) i `la` (w klamka ×3, kasztelan, klatki, latem, oddzielała); w drzwiami zniknęły d, rz, wi, a pojawił się `drzwi` (treningowe „drzwi” ×5). Przy 250 → 500 brak zmian: dodatkowe scalenia dotyczą innych słów korpusu.
- **Ocena hipotezy:** wspierana w częściach (a), (b), (c), (d); niewspierana w twierdzeniu o rdzeniu (4/8).

### 4.2 Word2Vec

- **Ranking zmienił się w 4 z 5 miejsc** (wspólny tylko `wytrych`). Po odfiltrowaniu słów funkcyjnych M1 ma 3 słowa związane z budowlą (chroniła, wieża, fosa) i 2 z mechanizmem (klucz, wytrych); M2 ma 4 słowa z mechanizmem (wytrych, klamka, drzwi, klucz) i 1 neutralne (były). Kierunek zgodny z oczekiwaniem.
- **wieża** spadła z miejsca 4 (cos 0,514) na 19 (0,320): jej częstość i liczba wspólnych zdań ze „zamek” spadły z 4 do 2. Zdanie „Odrestaurowany zamek przyciąga turystów, a jego wieża jest widoczna z całej okolicy.” jest tylko w V1; zostały „Król kazał wybudować zamek nad rzeką, żeby wieża obronna chroniła granicę.” i „Stary zamek stał na wzgórzu, a otaczały go mury, fosa i wysoka wieża.”
- **chroniła** nie ma wektora w M2: w V2 występuje raz („Król kazał wybudować zamek…”), a w V1 dwa razy (to zdanie i „Przewodnik opowiadał, jak wojska oblegały zamek, a fosa chroniła go przez całą zimę.”); w M2 jest poniżej `min_count`=2.
- **klucz wypadł z top-5 mimo wzrostu liczby wspólnych zdań ze „zamek” z 3 do 6**: cos spadło z 0,639 do 0,471 (miejsce 8). Podobieństwo nie rośnie więc wprost z liczbą współwystąpień. W V1 (przewaga zdań o budowli) `klucz` jest mimo to na 1. miejscu, co dla pojedynczego ziarna przeczy intuicji.
- **do, nie, z** w top-5 M2 to słowa funkcyjne o 2-3 wystąpieniach; „nie” ma 1 wystąpienie w V1 i 3 w V2 („…klucz nie pasował…”, „…klucz nie chciał…”, „…zamek nie ustąpił.”).
- **Test ziaren:** kierunek jest stabilny (10/10 w obu wariantach). Efekt jest asymetryczny: cos do G2 jest taki sam w obu wariantach (0,440), a cos do G1 spada z 0,475 do 0,357. Zmiana proporcji osłabia więc związek „zamek” ze słowami budowli, a nie wzmacnia związku z mechanizmem. W V1 przewaga G1 jest mała (+0,035), w V2 większa (−0,083).

### 4.3 PCA i grupy

- **PCA:** PC1 (35,9%) rozdziela grupy: wszystkie 5 słów G1 leży po prawej (PC1 > ok. 0,6), wszystkie 5 słów G2 po lewej (PC1 < ok. −0,9). „Zamek” leży między grupami (PC1 ≈ 0,2, PC2 ≈ −0,9), najbliżej (na wykresie) króla i klucza. Dwie składowe obejmują tylko 55,7% wariancji, więc odległości na wykresie są przybliżone.
- **Cosinusy potwierdzają grupowanie:** 0,615 i 0,703 wewnątrz grup wobec 0,355 między grupami.
- **Pary:** (mury, fosa) 0,852 - 2 wspólne zdania, np. „Fosa oddzielała zamek od miasta, więc rycerz musiał czekać na opuszczenie mostu, a mury były wysokie.” Grupy nie są jednak rozdzielne parami: najwyższa para międzygrupowa (mury, klamka, 0,576, 0 wspólnych zdań) jest bardziej podobna niż najniższa para wewnątrz grupy (rycerz, wieża, 0,473, 0 wspólnych zdań).

### 4.4 Uwaga

- **Z „zamek” w kluczach** największą wagę ma samo „zamek” (0,396 i 0,337), bo `q·q` jest największy; stąd cos(c, e) = 0,874 / 0,820 i cos(c_budowla, c_mechanizm) = 0,850 - różnica między zdaniami jest mała.
- **Tylko otoczenie:** wagi są prawie płaskie (A12: 0,059-0,132; B1: 0,059-0,105), a słowa funkcyjne („bo” 0,105, „a” 0,087, „nie” 0,088) mają wagi porównywalne ze słowami tematycznymi (klucz 0,104, drzwi 0,096). Bez uczonych `W_Q`, `W_K`, `W_V` uwaga niewiele wybiera.
- **Mimo to `c` zależy od zdania:** cos(c_budowla, c_mechanizm) = 0,607, a `c` jest bliższy centroidowi „własnej” grupy (A12: G1 0,770 vs G2 0,576; B1: G2 0,964 vs G1 0,559), podczas gdy statyczne `e(zamek)` jest w obu zdaniach to samo. Wynik częściowo wynika z konstrukcji: zdania zawierają słowa z grup (król, rycerz; klucz, drzwi, klamka, ślusarz), z których składa się `c`.

### 4.5 Ograniczenia

- Mały korpus (60 zdań, 571 tokenów, 364 formy), w wariantach Word2Vec 36 zdań; słowa grup mają 2-6 wystąpień w V1 i V2, dlatego wyniki sprawdzono dla 10 ziaren.
- Korpus sztuczny: zdania napisane pod eksperyment, każde zdanie ze „zamek” zawiera słowo tematyczne (16/16), co ułatwia rozdzielenie znaczeń; na naturalnym tekście efekt byłby słabszy.
- Brak lematyzacji (np. „drzwi” ≠ „drzwiach”).
- Word2Vec daje jeden wektor dla obu znaczeń; rozdzielenie w zdaniu wymaga reprezentacji kontekstowych (np. BERT).
- PCA 2D obejmuje 55,7% wariancji; w sekcji uwagi nieuczone projekcje i osobny model (`min_count=1`).
- BPE: 8 par, limity dobrane po przeglądzie pomocniczym; w głównym porównaniu Word2Vec jedno ziarno (wymóg zadania), stabilność oceniono osobno.

## 5. Wnioski

1. **BPE:** zwiększenie słownika ze 100 do 250 skróciło widziane formy z 2,75 do 1,00 tokenu/słowo, a niewidziane z 4,50 do 2,88; dalsze zwiększenie do 500 nic nie zmienia. `[UNK]` nie wystąpił.
2. **Rdzeń niewidzianych form** jest scalony w 4 z 8 przypadków. O podziale decyduje kolejność reguł z treningu: częste `zam` (ranga 3) rozbija rdzenie klucz- i rycerz-.
3. **Word2Vec:** zmiana proporcji 12:4 → 4:12 zmieniła 4 z 5 sąsiadów „zamek”: w M1 słowa budowli (wieża, fosa, chroniła), w M2 mechanizmu (wytrych, klamka, drzwi, klucz). Kierunek jest stabilny w 10/10 ziaren, ale efekt jest mały (+0,035 w V1) i wynika z osłabienia związku z G1 (0,475 → 0,357), nie ze wzmocnienia G2 (0,440 w obu).
4. **Grupy tematyczne:** PC1 (35,9%) rozdziela G1 i G2; średni cosinus 0,615 / 0,703 wewnątrz grup wobec 0,355 między nimi. Pojedyncze pary się nakładają (mury-klamka 0,576 > rycerz-wieża 0,473).
5. **Uwaga:** przy tym samym zapytaniu wektor kontekstu zależy od zdania (cos między `c` obu zdań 0,607 bez „zamek” w kluczach), ale bez uczonych projekcji wagi są prawie płaskie, a przy kluczach ze „zamek” dominuje własna pozycja (waga 0,34-0,40).
6. **Kierunki poprawy:** większy, naturalny korpus; lematyzacja; więcej ziaren; modele kontekstowe (np. HerBERT) do rozróżniania znaczeń „zamek”.

---

## Załącznik A - korpus (60 zdań)

**Pula A - zamek jako budowla** (1-12 w V1, 1-4 w V2):

1. Stary zamek stał na wzgórzu, a otaczały go mury, fosa i wysoka wieża.
2. Rycerz strzegł bramy, gdy król wjeżdżał na dziedziniec, a zamek wydawał się niezdobyty.
3. Król kazał wybudować zamek nad rzeką, żeby wieża obronna chroniła granicę.
4. Fosa oddzielała zamek od miasta, więc rycerz musiał czekać na opuszczenie mostu, a mury były wysokie.
5. Turyści zwiedzali zamek i fotografowali kamienne mury o zachodzie słońca.
6. Według legendy rycerz ukrył skarb w podziemiach, nad którymi stoi zamek.
7. Przewodnik opowiadał, jak wojska oblegały zamek, a fosa chroniła go przez całą zimę.
8. Odrestaurowany zamek przyciąga turystów, a jego wieża jest widoczna z całej okolicy.
9. Podczas turnieju rycerz walczył na dziedzińcu, a zamek wypełniał się widzami.
10. Dzieci z wycieczki chciały zobaczyć zamek, bo wieża podobno skrywa tajemnicę.
11. Historyk wyjaśnił, że zamek w czternastym wieku zbudował król.
12. Rycerz i król ucztowali w wielkiej sali, a zamek rozbrzmiewał muzyką.
13. Mury obronne i zamek przetrwały wojny dzięki grubym kamiennym blokom.
14. Z lotu ptaka zamek wygląda jak kamienna twierdza wśród zielonych wzgórz.
15. W sali tronowej zamek przechowuje dawne gobeliny i zbroje.
16. Mgła spowijała zamek tak szczelnie, że widać było tylko czubek wieży.
17. Kasztelan zarządzał całym majątkiem, a zamek był jego siedzibą.
18. Latem zamek tętni życiem, bo odbywają się tu festiwale historyczne.
19. Pod murami ciągną się lochy, a zamek ma też ukryte przejścia.
20. Król podarował zamek swojemu wiernemu rycerzowi w nagrodę za służbę.

**Pula B - zamek jako mechanizm** (1-4 w V1, 1-12 w V2):

1. Ślusarz wymienił zamek, bo klucz nie pasował do drzwi, a klamka luźno wisiała.
2. Wytrych złamał się w dziurce, więc zamek zablokował się na dobre, mimo że klucz był w kieszeni.
3. Drzwi i klamka były nowe, ale zamek zacinał się od początku.
4. Ślusarz wsunął wytrych do dziurki i zamek w końcu zaskoczył, więc klucz okazał się zbędny.
5. Drzwi wejściowe miały zamek, który zacinał się przy każdym użyciu.
6. Klamka poruszała się swobodnie, lecz zamek nie ustąpił.
7. Ślusarz polecił zamek antywłamaniowy ze wzmocnioną wkładką.
8. Wymieniliśmy zamek po przeprowadzce, bo każdy z lokatorów miał własny klucz.
9. Zamek w szafce był zardzewiały, więc klucz obracał się z oporem.
10. Ślusarz przyjechał po godzinie i szybko otworzył zamek bez uszkodzeń.
11. Drzwi biurowe otwiera zamek elektroniczny na kartę lub kod.
12. Zimą zamek w samochodzie zamarzł i klucz nie chciał się obrócić.
13. Zamek szyfrowy w walizce ma trzy pokrętła z cyframi.
14. Dorobiłem drugi klucz, bo zamek w mieszkaniu często się zacinał.
15. Mieszkańcy zamontowali zamek domofonowy przy wejściu do klatki.
16. Drzwi do piwnicy mają zamek, który otwiera tylko administrator.
17. Zamek w rowerowej kłódce wymaga ostrożnego smarowania.
18. Uszkodzony zamek w furtce uniemożliwiał jej zamknięcie.
19. Mechanik sprawdził zamek i stwierdził, że trzeba wymienić sprężynę.
20. Nowy zamek w sejfie ma dwa niezależne zabezpieczenia.

**Pula C - bez „zamek”** (w obu wariantach):

1. Sieć neuronowa uczy się rozpoznawać wzorce w danych.
2. Model języka przetwarza tekst i przewiduje kolejne słowa.
3. Embedding reprezentuje słowo jako wektor liczb.
4. Podobne słowa mają podobne wektory w przestrzeni embeddingów.
5. Tokenizacja dzieli tekst na mniejsze jednostki zwane tokenami.
6. Mechanizm uwagi waży znaczenie różnych słów w zdaniu.
7. Rozpoznawanie mowy zamienia sygnał dźwiękowy na tekst.
8. Synteza mowy generuje dźwięk na podstawie tekstu.
9. Kot siedzi na macie i mruczy cicho.
10. Pies biega po ogrodzie i szczeka głośno.
11. Zwierzęta domowe potrzebują opieki i jedzenia.
12. Kot i pies lubią spać na ciepłej kanapie.
13. Sieć rekurencyjna przetwarza sekwencje krok po kroku.
14. Wektor kontekstu podsumowuje znaczenie całego zdania.
15. Model uczy się na dużym zbiorze danych tekstowych.
16. Język polski ma bogatą fleksję i odmianę wyrazów.
17. Koń galopuje po łące, a chomik biega w kołowrotku.
18. Odmiana czasownika zależy od osoby, liczby i czasu.
19. Papuga powtarza słowa, które często słyszy w domu.
20. Wektor słowa zmienia się podczas treningu modelu.

---

Prowadzący: dr Tomasz Cieplak / mgr inż. Patryk Marek, Rok akademicki: 2026/27
