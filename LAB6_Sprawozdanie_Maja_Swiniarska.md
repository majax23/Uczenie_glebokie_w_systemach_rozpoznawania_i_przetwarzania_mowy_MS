# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy

**Zajęcia nr / temat:** 6 / Projekt generacyjny, cz. 1 – przygotowanie danych 

**Autor / grupa:** Maja Świniarska GL.04

**Data:** 19.11.2026

## 1. Cel ćwiczenia

Celem było przygotowanie danych z polskiego korpusu do zadania przewidywania następnego znaku, które posłużą na laboratorium 7 do treningu modelu Char-RNN. 

## 2. Metoda i implementacja

### Narzędzia

Python, PyTorch 2.14.1 (`Dataset`, `DataLoader`), NumPy, pandas, matplotlib; środowisko Google Colab. Wszystkie ustawienia są zebrane w jednej komórce, a ziarno losowania jest ustalone (`SEED = 0`).

### Korpus

| | |
|---|---|
| Utwór | Henryk Sienkiewicz, *Quo vadis* (cała powieść, 1896) |
| Źródło | Wolne Lektury – https://wolnelektury.pl/katalog/lektura/quo-vadis/, plik TXT: https://wolnelektury.pl/media/book/txt/quo-vadis.txt |
| Podstawa tekstu | H. Sienkiewicz, *Quo vadis*, PIW, Warszawa 1976 |
| Warunki użycia | utwór w domenie publicznej (autor zmarł w 1916 r.); korzystanie zgodnie z „Zasadami wykorzystania Wolnych Lektur” |

**Uzasadnienie wyboru.** Powieść jest duża (ok. 1,09 mln znaków po oczyszczeniu), więc nawet 10-procentowe części walidacyjna i testowa mają ponad 100 tys. znaków. Tekst pochodzi od jednego autora, ma jednolity styl i otwartą licencję.

**Charakterystyka tekstu.** Powieść historyczna o Rzymie za panowania Nerona i prześladowaniach pierwszych chrześcijan. Zawiera dużo narracji i dialogów (myślniki `—`, cudzysłowy `„”`, `«»`), a język jest stylizowany na archaiczny (np. *rzekłszy*, *bom sam widział*, *Niewielem*). Pojawiają się imiona łacińskie i greckie (Petroniusz, Winicjusz, Nero). W tekście nie ma cyfr.

**Przygotowanie tekstu** (funkcja `oczysc`):

1. normalizacja Unicode do NFC,
2. ujednolicenie końców linii `\r\n` → `\n`,
3. usunięcie stopki Wolnych Lektur (od linii `-----` do końca pliku: licencja, informacje o wydaniu, ISBN),
4. usunięcie nagłówka (autor, tytuł, ISBN): tekst właściwy zaczyna się od „Rozdział pierwszy”,
5. usunięcie ewentualnych znaczników HTML (nie wystąpiły),
6. usunięcie spacji na końcach linii, zwinięcie wielokrotnych spacji do jednej i 3+ pustych linii do jednej pustej.

Wielkość liter, interpunkcja i podział na akapity zostały zachowane, bo niosą informację o stylu. Poprawność czyszczenia sprawdzają asercje: w tekście nie ma „wolnelektury”, „ISBN”, `\r` ani znaczników HTML, a tekst jest w postaci NFC.


**Tabela 1.** Tekst przed i po czyszczeniu

| miara | surowy | po czyszczeniu |
|---|---:|---:|
| znaków | 1 097 217 | 1 086 581 |
| znaków `\r` | 8 853 | 0 |
| znaczników HTML | 0 | 0 |
| polskich liter | 82 181 | 82 149 |
| linii | 8 854 | 8 661 |
| akapitów | 4 327 | 4 316 |

Usunięto 10 636 znaków (ok. 1%): 8 853 znaki `\r`, a resztę stanowią nagłówek i stopka. Polskich liter ubyło tylko 32, czyli tyle, ile było ich w nagłówku i stopce. Polskie znaki odczytano poprawnie (UTF-8): np. ł – 24 845, ę – 13 348, ą – 10 497 wystąpień.

**Fragment po oczyszczeniu (początek tekstu):**

> Rozdział pierwszy
>
> Petroniusz obudził się zaledwie koło południa i jak zwykle, zmęczony bardzo. Poprzedniego dnia był na uczcie u Nerona, która przeciągnęła się do późna w noc. Od pewnego czasu zdrowie jego zaczęło się psuć.


### Podział danych

Korpus to jeden długi tekst, dlatego wydzielono trzy rozłączne, ciągłe fragmenty w kolejności tekstu: 80% trening / 10% walidacja / 10% test. Granice przesunięto do najbliższego końca akapitu (`\n\n`), żeby nie ciąć zdań. Granice wypadły na znakach 869 557 i 979 160. Asercja potwierdza, że połączone fragmenty odtwarzają cały tekst.

Podział wykonano przed tworzeniem okien, a okna powstają osobno wewnątrz każdego fragmentu. Sąsiednie okna pokrywają się w `SEKW − 1` znakach, więc losowy podział gotowych okien wprowadziłby niemal identyczne przykłady do treningu i walidacji (przeciek danych).

**Uzasadnienie proporcji 80/10/10.** Model potrzebuje jak najwięcej tekstu do nauki, a przy ponad 1 mln znaków 10% to wciąż ok. 110 tys. znaków, czyli ponad 100 tys. par na walidację i test. To wystarcza do stabilnej oceny straty i z dużym zapasem wystarcza dla największego badanego `SEKW = 160`. Część testowa to końcowe rozdziały powieści, więc test sprawdza też generalizację na dalszą część fabuły.

Podział zapisano do plików `quovadis_trening.txt`, `quovadis_walidacja.txt` i `quovadis_test.txt`, a słownik do `slownik.json`.

<img width="644" height="209" alt="image" src="https://github.com/user-attachments/assets/f0688db6-84d2-497f-a16b-784cbc1ca832" />


*Rys. 1. Podział korpusu na ciągłe fragmenty.*

### Słownik i kodowanie

Słownik zbudowano wyłącznie na części treningowej (`sorted(set(trening))`) i zastosowano do wszystkich zbiorów. Na końcu słownika zarezerwowano dodatkowe ID dla znaku nieznanego `UNK` (`'�'`). Znak spoza słownika jest kodowany jako `UNK`, a liczba takich wystąpień jest zliczana.


### Dataset, DataLoader i konfiguracja

Para to okno `SEKW` znaków (wejście `x`) oraz to samo okno przesunięte o jeden znak (cel `y`). Okno przesuwa się z krokiem `KROK = 1`. Klasa zwraca `max(0, …)` par, więc zbyt krótki fragment daje 0 przykładów zamiast błędu.


**Tabela 2.** Konfiguracja

| parametr | wartość |
|---|---|
| podział | 80 / 10 / 10 (ciągłe fragmenty, granice na końcu akapitu) |
| rozmiar słownika `V` | 83 (82 znaki z treningu + `UNK`) |
| długość okna `SEKW` | 40 (eksperyment: 20, 40, 80, 160) |
| krok okna `KROK` | 1 |
| `batch_size` | 32 |
| trening | `shuffle=True`, `drop_last=True`, generator z ziarnem 0 |
| walidacja, test | `shuffle=False`, `drop_last=False` (ocena obejmuje wszystkie pary) |
| typ tensorów | `torch.int64`, kształt `[B, SEKW]` |

## 3. Wyniki


### Liczności zbiorów

**Tabela 3.** Statystyki zbiorów (`SEKW = 40`, `BATCH = 32`)

| zbiór | znaków | % korpusu | różnych znaków | wystąpień spoza słownika | par | batchy |
|---|---:|---:|---:|---:|---:|---:|
| trening | 869 557 | 80,03 | 82 | 0 | 869 517 | 27 172 |
| walidacja | 109 603 | 10,09 | 73 | 0 | 109 563 | 3 424 |
| test | 107 421 | 9,89 | 77 | 3 | 107 381 | 3 356 |
| razem | 1 086 581 | 100 | – | 3 | 1 086 461 | 33 952 |

Proporcje odbiegają od 80/10/10 o mniej niż 0,1 punktu procentowego z powodu przesunięcia granic do końca akapitu. Liczba par w każdym zbiorze to dokładnie `N − SEKW`.

W treningu `drop_last=True` odrzuca 13 par (869 517 − 27 172 · 32). W walidacji i teście ostatni batch jest niepełny: ma 27 par w walidacji (109 563 − 3 423 · 32) i 21 w teście (107 381 − 3 355 · 32). Dzięki temu ocena obejmuje wszystkie pary.

### Kontrola danych

**Słownik.** `V = 83`. Pełna zawartość słownika:

```
"\n !'(),-.:;>?ABCDEFGHIJKLMNOPRSTUVWZabcdefghijklmnopqrstuvwxyz«»ÓóąćęŁłńŚśŹźŻż—”„…�"
```

**Test kodowania i dekodowania** (fragment treningu od znaku 1000):

```
Fragment  : 'e słynny wspólnik Sewerusa, Celer, rozszerzył mu, przebudował i urządził z tak n'
ID        : [40, 1, 54, 70, 60, 49, 49, 60, 1, 58, 54, 51, 65, 47, 49, 44, 46, 1, 30, 40, ...]
Dekodowany: 'e słynny wspólnik Sewerusa, Celer, rozszerzył mu, przebudował i urządził z tak n'
```

Dekodowanie odtwarza fragment, a dodatkowa asercja `dec(enc(trening)) == trening` potwierdza, że odtwarza cały zbiór treningowy.

**Znaki spoza słownika.** W walidacji nie ma takich znaków. W teście jest jeden: wielka litera `Q` (U+0051), 3 wystąpienia. Pochodzi z łacińskiego zwrotu *„Quo vadis, Domine?”*, który pada dopiero w końcowych rozdziałach powieści. W treningu wielkie `Q` się nie pojawia, bo tytuł z nagłówka został usunięty. Zastosowana reguła: znak spoza słownika → `UNK`. Obejmuje ona 3 znaki, czyli 0,003% testu. Test dokładnego odtworzenia tekstu wykonano na treningu, który jest w całości pokryty przez słownik.

**Kontrola batchy.** Z każdego loadera pobrano jeden batch i sprawdzono asercjami: kształt, typ, zakres ID w przedziale `[0, V − 1]` oraz przesunięcie `x[:, 1:] == y[:, :-1]` dla wszystkich wierszy batcha.

**Tabela 4.** Wynik kontroli batchy

| zbiór | kształt X | kształt Y | dtype | min ID | max ID | V − 1 | przesunięcie | para = źródło |
|---|---|---|---|---:|---:|---:|---|---|
| trening | (32, 40) | (32, 40) | torch.int64 | 0 | 80 | 82 | ✓ | ✓ |
| walidacja | (32, 40) | (32, 40) | torch.int64 | 1 | 70 | 82 | ✓ | ✓ |
| test | (32, 40) | (32, 40) | torch.int64 | 1 | 77 | 82 | ✓ | ✓ |

**Sprawdzone pary wejście–cel.** Treningowy loader tasuje przykłady, dlatego z każdego zbioru pobrano też parę o znanym indeksie (środek zbioru). Jej ID porównano z zakodowanym tekstem źródłowym, a ostatni znak celu bezpośrednio ze znakiem korpusu na pozycji `s + SEKW`:

| zbiór | indeks pary | wejście `x` | cel `y` | ostatni znak celu = źródło |
|---|---:|---|---|---|
| trening | 434 758 | `'e i w tej jednej izbie wśród prostaków m'` | `' i w tej jednej izbie wśród prostaków mi'` | `'i'` = `'i'` ✓ |
| walidacja | 54 781 | `' rzekłszy zmarszczył brwi, lecz począł p'` | `'rzekłszy zmarszczył brwi, lecz począł pa'` | `'a'` = `'a'` ✓ |
| test | 53 690 | `' Góry Albańskie, które ukazały się cudne'` | `'Góry Albańskie, które ukazały się cudne,'` | `','` = `','` ✓ |

### Najczęstsze znaki w treningu

<img width="920" height="395" alt="image" src="https://github.com/user-attachments/assets/3ddd1f7f-c622-4297-8132-d5704defb1d5" />


*Rys. 2. 30 najczęstszych znaków w zbiorze treningowym (␣ – spacja, \n – nowa linia).*

**Tabela 5.** 10 najczęstszych znaków w treningu

| znak | ␣ | i | a | e | o | z | n | s | c | w |
|---|---|---|---|---|---|---|---|---|---|---|
| % treningu | 15,50 | 7,62 | 6,37 | 6,09 | 5,41 | 5,13 | 3,80 | 3,30 | 3,26 | 3,18 |

### Wpływ długości okna `SEKW`

Ten sam tekst, ten sam podział i ten sam krok (`KROK = 1`); zmieniana była tylko długość okna.

**Tabela 6.** Liczba par w zależności od `SEKW`

| SEKW | trening | walidacja | test | ubytek par względem N (trening / walidacja / test) |
|---:|---:|---:|---:|---|
| 20 | 869 537 | 109 583 | 107 401 | 0,002% / 0,018% / 0,019% |
| 40 | 869 517 | 109 563 | 107 381 | 0,005% / 0,036% / 0,037% |
| 80 | 869 477 | 109 523 | 107 341 | 0,009% / 0,073% / 0,074% |
| 160 | 869 397 | 109 443 | 107 261 | 0,018% / 0,146% / 0,149% |

<img width="1288" height="377" alt="image" src="https://github.com/user-attachments/assets/eda8383b-7c04-42a3-9be2-399161c89c10" />


*Rys. 3. Liczba par wejście–cel w zależności od `SEKW` w trzech zbiorach. Każdy panel ma własną skalę osi Y.*


### Dodatkowo: najczęstsze słowa

W treningu jest 135 190 słów (tokenów `\w+` po zamianie na małe litery) i 22 539 różnych form.

<img width="937" height="421" alt="image" src="https://github.com/user-attachments/assets/82197af7-9dd5-4cb6-aaad-6718154c0342" />


*Rys. 4. 25 najczęstszych słów w zbiorze treningowym.*

## 4. Analiza i interpretacja

Czyszczenie usunęło tylko nagłówek, stopkę i znaki `\r`, co potwierdzają asercje i podgląd tekstu. Dzięki temu model nie będzie generował licencji ani adresów URL. Walidacja i test są prawie w całości pokryte przez słownik. Wyjątek to 3 wystąpienia `Q` w teście, zamienione na `UNK`, co ma pomijalny wpływ na ocenę. Ograniczenie: w słowniku jest nietypowy znak `>`, prawdopodobnie artefakt pliku źródłowego.

Batche mają kształt `[32, 40]`, typ `int64` i ID w zakresie słownika, a przesunięcie celu jest poprawne. Porównanie par z tekstem źródłowym, łącznie z ostatnim znakiem celu, potwierdza, że cele to rzeczywiście następne znaki korpusu.

Rozkład jest silnie nierównomierny: spacja to 15,5% znaków, dalej samogłoski `i`, `a`, `e`, `o`. Model przewidujący zawsze spację miałby trafność ok. 15,5%. Rzadkie znaki (wielkie litery, `«»`, `Ź`) model nauczy się słabiej.

Liczba par to `N − SEKW`, więc maleje liniowo, ale pomijalnie: przy `SEKW = 160` ubytek to 0,02% w treningu i ok. 0,15% w walidacji i teście. Dłuższe okno daje za to więcej kontekstu (40 znaków to ok. 6–7 słów, 160 to 1–2 zdania) i proporcjonalnie zwiększa koszt epoki (ok. 34,8 mln pozycji przy `SEKW = 40` i 139 mln przy 160). Dla RNN trudniejsze jest też wykorzystanie odległej informacji.

Dominują słowa funkcyjne (*i*, *się*, *w*, *nie*), a jedyne słowo treściowe w pierwszej 25 to *Winicjusz*. 22,5 tys. różnych form przy 135 tys. tokenów to efekt fleksji polszczyzny i argument za modelem znakowym.

## 5. Wnioski

1. *Quo vadis* (ok. 1,09 mln znaków po oczyszczeniu) to wystarczająco duży i jednolity korpus do treningu Char-RNN.
2. Podział 80/10/10 na ciągłe fragmenty przed tworzeniem okien zapobiega przeciekowi danych. Każdy zbiór ma ponad 100 tys. par.
3. Słownik z treningu (`V = 83`) pokrywa prawie cały tekst, a kodowanie i dekodowanie odtwarza go bezbłędnie. Batche i pary wejście–cel przeszły wszystkie kontrole.
4. `SEKW` praktycznie nie zmienia liczby par, ale zwiększa kontekst i koszt epoki. `SEKW = 40` to rozsądny punkt startowy.

Podział (`quovadis_*.txt`) i słownik (`slownik.json`) zostały zapisane do użycia na kolejnych zajęciach.


