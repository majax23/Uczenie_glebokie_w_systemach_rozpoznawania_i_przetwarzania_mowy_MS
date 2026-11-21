# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 6 / Projekt generacyjny, cz. 1 – przygotowanie danych dla Char-RNN
**Autor / grupa:** Maja Świniarska / …
**Data:** 21.11.2026

## 1. Cel ćwiczenia

Celem było przygotowanie danych z polskiego korpusu do zadania przewidywania następnego znaku, które posłużą na laboratorium 7 do treningu modelu Char-RNN. Zakres pracy:

- wybór, oczyszczenie i opis korpusu,
- podział tekstu na rozłączne części treningową, walidacyjną i testową **przed** utworzeniem okien,
- zbudowanie słownika znaków na części treningowej i obsługa znaków spoza słownika,
- utworzenie `Dataset` i `DataLoader` dla każdego zbioru (pary wejście–cel z przesunięciem o 1 znak) oraz kontrola batchy,
- zbadanie, jak długość okna `SEKW` wpływa na liczbę par w każdym zbiorze.

## 2. Metoda i implementacja

### 2.1 Narzędzia

Python, PyTorch 2.14.1 (`Dataset`, `DataLoader`), NumPy, pandas, matplotlib; środowisko Google Colab. Cały kod znajduje się w notatniku `LAB6_Zadanie_Maja_Swiniarska.ipynb`. Wszystkie ustawienia są zebrane w jednej komórce, a ziarno losowania jest ustalone (`SEED = 0`).

### 2.2 Korpus

| | |
|---|---|
| Utwór | Henryk Sienkiewicz, *Quo vadis* (cała powieść, 1896) |
| Źródło | Wolne Lektury – https://wolnelektury.pl/katalog/lektura/quo-vadis/, plik TXT: https://wolnelektury.pl/media/book/txt/quo-vadis.txt |
| Podstawa tekstu | H. Sienkiewicz, *Quo vadis*, PIW, Warszawa 1976 (wg stopki pliku) |
| Warunki użycia | utwór w domenie publicznej (autor zmarł w 1916 r.); korzystanie zgodnie z „Zasadami wykorzystania Wolnych Lektur” |

**Uzasadnienie wyboru.** Powieść jest duża (ok. 1,09 mln znaków po oczyszczeniu, więcej niż I tom „Lalki” z ćwiczenia), więc nawet 10-procentowe części walidacyjna i testowa mają ponad 100 tys. znaków. Tekst pochodzi od jednego autora i ma jednolity styl, ma otwartą licencję i ten sam format co plik z ćwiczenia, co ułatwia odtworzenie potoku.

**Charakterystyka tekstu.** Powieść historyczna o Rzymie za panowania Nerona i prześladowaniach pierwszych chrześcijan. Zawiera dużo narracji i dialogów (myślniki `—`, cudzysłowy `„”`, `«»`), a język jest stylizowany na archaiczny (np. *rzekłszy*, *bom sam widział*, *Niewielem*). Pojawiają się imiona łacińskie i greckie (Petroniusz, Winicjusz, Nero). W tekście nie ma cyfr.

**Przygotowanie tekstu** (funkcja `oczysc`):

1. normalizacja Unicode do NFC,
2. ujednolicenie końców linii `\r\n` → `\n`,
3. usunięcie stopki Wolnych Lektur (od linii `-----` do końca pliku: licencja, informacje o wydaniu, ISBN),
4. usunięcie nagłówka (autor, tytuł, ISBN): tekst właściwy zaczyna się od „Rozdział pierwszy”,
5. usunięcie ewentualnych znaczników HTML (nie wystąpiły),
6. usunięcie spacji na końcach linii, zwinięcie wielokrotnych spacji do jednej i 3+ pustych linii do jednej pustej.

Wielkość liter, interpunkcja i podział na akapity zostały zachowane, bo niosą informację o stylu. Poprawność czyszczenia sprawdzają asercje: w tekście nie ma „wolnelektury”, „ISBN”, `\r` ani znaczników HTML, a tekst jest w postaci NFC.

```python
def oczysc(t):
    t = unicodedata.normalize('NFC', t)
    t = t.replace('\r\n', '\n').replace('\r', '\n')
    if '\n-----' in t:                              # stopka serwisu
        t = t[:t.rfind('\n-----')]
    m = re.search(r'ISBN[^\n]*\n', t)               # nagłówek: autor, tytuł, ISBN
    if m:
        t = t[m.end():]
    t = re.sub(r'<[^>]+>', '', t)                   # znaczniki HTML
    t = re.sub(r'[ \t]+\n', '\n', t)                # spacje na końcu linii
    t = re.sub(r'[ \t]{2,}', ' ', t)                # wielokrotne spacje
    t = re.sub(r'\n{3,}', '\n\n', t)                # wielokrotne puste linie
    return t.strip()
```

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

Koniec tekstu to ostatnie zdanie powieści (*„Quo vadis, Domine?”*), czyli stopka została usunięta w całości.

### 2.3 Podział danych

Korpus to jeden długi tekst, dlatego wydzielono trzy **rozłączne, ciągłe fragmenty** w kolejności tekstu: **80% trening / 10% walidacja / 10% test**. Granice przesunięto do najbliższego końca akapitu (`\n\n`), żeby nie ciąć zdań. Granice wypadły na znakach 869 557 i 979 160. Asercja potwierdza, że połączone fragmenty odtwarzają cały tekst.

Podział wykonano **przed** tworzeniem okien, a okna powstają osobno wewnątrz każdego fragmentu. Sąsiednie okna pokrywają się w `SEKW − 1` znakach, więc losowy podział gotowych okien wprowadziłby niemal identyczne przykłady do treningu i walidacji (przeciek danych).

**Uzasadnienie proporcji 80/10/10.** Model potrzebuje jak najwięcej tekstu do nauki, a przy ponad 1 mln znaków 10% to wciąż ok. 110 tys. znaków, czyli ponad 100 tys. par na walidację i test. To wystarcza do stabilnej oceny straty i z dużym zapasem wystarcza dla największego badanego `SEKW = 160`. Część testowa to końcowe rozdziały powieści, więc test sprawdza też generalizację na dalszą część fabuły.

Podział zapisano do plików `quovadis_trening.txt`, `quovadis_walidacja.txt` i `quovadis_test.txt`, a słownik do `slownik.json`. Zostaną użyte bez zmian na laboratoriach 7 i 8.

![Podział korpusu](podzial_korpusu.png)

*Rys. 1. Podział korpusu na ciągłe fragmenty.*

### 2.4 Słownik i kodowanie

Słownik zbudowano **wyłącznie na części treningowej** (`sorted(set(trening))`) i zastosowano do wszystkich zbiorów. Na końcu słownika zarezerwowano dodatkowe ID dla znaku nieznanego `UNK` (`'�'`). Znak spoza słownika jest kodowany jako `UNK`, a liczba takich wystąpień jest zliczana.

```python
znaki = sorted(set(FRAG['trening']))
znaki.append(UNK)                         # ostatnie ID = znak nieznany
stoi = {z: i for i, z in enumerate(znaki)}
itos = {i: z for z, i in stoi.items()}

def enc(s):  return [stoi.get(c, stoi[UNK]) for c in s]
def dec(ids): return ''.join(itos[i] for i in ids)
```

### 2.5 Dataset, DataLoader i konfiguracja

Para to okno `SEKW` znaków (wejście `x`) oraz to samo okno przesunięte o jeden znak (cel `y`). Okno przesuwa się z krokiem `KROK = 1`. Klasa zwraca `max(0, …)` par, więc zbyt krótki fragment daje 0 przykładów zamiast błędu.

```python
class DatasetZnakowy(Dataset):
    def __init__(self, dane, sekw, krok=1):
        self.dane = dane; self.sekw = sekw; self.krok = krok
    def __len__(self):
        return max(0, (len(self.dane) - self.sekw - 1) // self.krok + 1)
    def __getitem__(self, i):
        s = i * self.krok
        x = self.dane[s:s+self.sekw]
        y = self.dane[s+1:s+self.sekw+1]   # przesunięcie o 1
        return torch.tensor(x), torch.tensor(y)
```

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

Na tym etapie nie trenuje się modelu, więc metryki jakości (strata, perplexity) i porównanie z baseline'em pojawią się na laboratorium 7. Wynikami tego laboratorium są liczności zbiorów, wyniki kontroli danych i eksperyment z `SEKW`. Jako punkt odniesienia służy notatnik ćwiczeniowy (I tom „Lalki”).

### 3.1 Liczności zbiorów

**Tabela 3.** Statystyki zbiorów (`SEKW = 40`, `BATCH = 32`)

| zbiór | znaków | % korpusu | różnych znaków | wystąpień spoza słownika | par | batchy |
|---|---:|---:|---:|---:|---:|---:|
| trening | 869 557 | 80,03 | 82 | 0 | 869 517 | 27 172 |
| walidacja | 109 603 | 10,09 | 73 | 0 | 109 563 | 3 424 |
| test | 107 421 | 9,89 | 77 | 3 | 107 381 | 3 356 |
| razem | 1 086 581 | 100 | – | 3 | 1 086 461 | 33 952 |

Proporcje odbiegają od 80/10/10 o mniej niż 0,1 punktu procentowego z powodu przesunięcia granic do końca akapitu. Liczba par w każdym zbiorze to dokładnie `N − SEKW`.

W treningu `drop_last=True` odrzuca 13 par (869 517 − 27 172 · 32). W walidacji i teście ostatni batch jest niepełny: ma 27 par w walidacji (109 563 − 3 423 · 32) i 21 w teście (107 381 − 3 355 · 32). Dzięki temu ocena obejmuje wszystkie pary.

### 3.2 Kontrola danych

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

**Znaki spoza słownika.** W walidacji nie ma takich znaków. W teście jest jeden: wielka litera **`Q`** (U+0051), 3 wystąpienia. Pochodzi z łacińskiego zwrotu *„Quo vadis, Domine?”*, który pada dopiero w końcowych rozdziałach powieści. W treningu wielkie `Q` się nie pojawia, bo tytuł z nagłówka został usunięty. Zastosowana reguła: znak spoza słownika → `UNK`. Obejmuje ona 3 znaki, czyli 0,003% testu. Test dokładnego odtworzenia tekstu wykonano na treningu, który jest w całości pokryty przez słownik.

**Kontrola batchy.** Z każdego loadera pobrano jeden batch i sprawdzono asercjami: kształt, typ, zakres ID w przedziale `[0, V − 1]` oraz przesunięcie `x[:, 1:] == y[:, :-1]` dla **wszystkich** wierszy batcha.

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

### 3.3 Najczęstsze znaki w treningu

![Najczęstsze znaki](czestosc_znakow.png)

*Rys. 2. 30 najczęstszych znaków w zbiorze treningowym (␣ – spacja, \n – nowa linia).*

**Tabela 5.** 10 najczęstszych znaków w treningu

| znak | ␣ | i | a | e | o | z | n | s | c | w |
|---|---|---|---|---|---|---|---|---|---|---|
| % treningu | 15,50 | 7,62 | 6,37 | 6,09 | 5,41 | 5,13 | 3,80 | 3,30 | 3,26 | 3,18 |

### 3.4 Wpływ długości okna `SEKW`

Ten sam tekst, ten sam podział i ten sam krok (`KROK = 1`); zmieniana była tylko długość okna.

**Tabela 6.** Liczba par w zależności od `SEKW`

| SEKW | trening | walidacja | test | ubytek par względem N (trening / walidacja / test) |
|---:|---:|---:|---:|---|
| 20 | 869 537 | 109 583 | 107 401 | 0,002% / 0,018% / 0,019% |
| 40 | 869 517 | 109 563 | 107 381 | 0,005% / 0,036% / 0,037% |
| 80 | 869 477 | 109 523 | 107 341 | 0,009% / 0,073% / 0,074% |
| 160 | 869 397 | 109 443 | 107 261 | 0,018% / 0,146% / 0,149% |

![Liczba par vs SEKW](pary_vs_sekw.png)

*Rys. 3. Liczba par wejście–cel w zależności od `SEKW` w trzech zbiorach. Każdy panel ma własną skalę osi Y.*

### 3.5 Porównanie z notatnikiem ćwiczeniowym

| | ćwiczenie („Lalka”, t. I) | zadanie („Quo vadis”) |
|---|---|---|
| znaków w korpusie | 784 082 (bez czyszczenia) | 1 086 581 (po czyszczeniu) |
| słownik | 103 znaki, z całego tekstu | 83 (82 + `UNK`), tylko z treningu |
| podział | brak (jeden zbiór) | trening / walidacja / test |
| par przy SEKW = 40 | 784 042 | 869 517 / 109 563 / 107 381 |

### 3.6 Dodatkowo: najczęstsze słowa

W treningu jest 135 190 słów (tokenów `\w+` po zamianie na małe litery) i 22 539 różnych form.

![Najczęstsze słowa](czestosc_slow.png)

*Rys. 4. 25 najczęstszych słów w zbiorze treningowym.*

## 4. Analiza i interpretacja

**Korpus i czyszczenie.** Czyszczenie usunęło tylko elementy spoza utworu, co potwierdzają asercje i podgląd początku i końca tekstu. Gdyby w danych zostały licencja, adresy URL i numer ISBN, model mógłby je odtwarzać przy generowaniu. Ujednolicenie końców linii ma dodatkowy skutek: w ćwiczeniu `\r` był osobnym znakiem słownika i każdy koniec linii zajmował dwa kroki sekwencji, a tutaj zajmuje jeden.

**Słownik.** Słownik „Quo vadis” (83 pozycje) jest mniejszy niż w ćwiczeniu (103). Powody: brak `\r`, brak cyfr w tekście i budowanie słownika tylko na treningu. Mniejszy słownik oznacza mniejszą warstwę embeddingów i warstwę wyjściową modelu.

Walidacja (73 znaki) i test (77) mają mniej różnych znaków niż trening (82), a prawie wszystkie są pokryte przez słownik treningowy. Jedyny wyjątek to `Q` w teście. To pokazuje, dlaczego słownik musi pochodzić z treningu: model nie może znać znaków, których nie widział podczas uczenia, a obsługa `UNK` zapobiega błędowi przy kodowaniu. Przy 3 wystąpieniach wpływ na ocenę jest pomijalny.

Ograniczenie: w słowniku jest znak `>`, nietypowy dla prozy, prawdopodobnie artefakt redakcyjny pliku źródłowego. Nie był usuwany, bo nie należy do żadnej z rozpoznanych kategorii artefaktów (nagłówek, stopka, HTML).

**Kontrola danych.** Wszystkie batche mają oczekiwany kształt `[32, 40]` i typ `int64`, a ID mieszczą się w zakresie słownika. Przesunięcie celu jest poprawne dla każdego wiersza. Porównanie par o znanym indeksie z tekstem źródłowym, łącznie z ostatnim znakiem celu, którego warunek `x[1:] == y[:-1]` nie obejmuje, potwierdza, że cele odpowiadają rzeczywiście następnym znakom korpusu. Błąd przesunięcia mógłby dać tensory o poprawnym kształcie i zakresie, a mimo to błędnej treści.

**Częstość znaków.** Rozkład jest silnie nierównomierny. Spacja stanowi 15,5% znaków, czyli średnia długość słowa wynosi ok. 5–6 znaków. Kolejne miejsca zajmują samogłoski `i`, `a`, `e`, `o` oraz `z`, typowe dla polszczyzny (dwuznaki *sz*, *cz*, *rz*). Z polskich liter najczęstsze są `ł` (ok. 2,2%), `ę` i `ą`, więc model dostanie dużo przykładów polskich znaków diakrytycznych.

Ta nierównowaga oznacza, że naiwny model przewidujący zawsze spację osiągnąłby trafność ok. 15,5%. Jest to dolny punkt odniesienia dla oceny Char-RNN na laboratorium 7. Wielkie litery i rzadkie znaki (`«»`, `Ź`) mają bardzo mało przykładów, więc model nauczy się ich słabiej.

**Wpływ `SEKW`.** Przy kroku 1 liczba par to dokładnie `N − SEKW`, więc maleje liniowo, o 1 parę na każdy znak wydłużenia okna (rys. 3). W skali korpusu ten spadek jest pomijalny: nawet przy `SEKW = 160` ubytek wynosi 0,02% w treningu i ok. 0,15% w walidacji i teście. Względnie większy jest w krótszych zbiorach, bo każdy fragment traci tę samą liczbę `SEKW` par. Wydłużenie okna nie zmniejsza więc zauważalnie liczby przykładów.

Długość okna zmienia za to dwie inne rzeczy. Po pierwsze dłuższe okno daje modelowi więcej kontekstu: 40 znaków to ok. 6–7 słów, 160 znaków to już ok. 1–2 zdania. Po drugie rośnie koszt epoki: okna się pokrywają, więc łączna liczba pozycji przetwarzanych w epoce treningowej to ok. `N · SEKW`. Daje to ok. 34,8 mln pozycji dla `SEKW = 40` i ok. 139 mln dla `SEKW = 160`, czyli przy tej samej liczbie batchy każdy batch jest 4 razy dłuższy. Dodatkowo dłuższe sekwencje utrudniają RNN wykorzystanie odległej informacji. Ostateczny wybór `SEKW` wymaga pomiaru czasu epoki i straty walidacyjnej na laboratorium 7.

**Najczęstsze słowa (dodatkowo).** Rozkład słów jest bardzo skośny, zgodnie z prawem Zipfa. Czołówkę tworzą słowa funkcyjne (*i* – ok. 5,4 tys. wystąpień, dalej *się*, *w*, *nie*, *na*, *z*, *że*). Jedynym słowem treściowym w pierwszej 25 jest imię głównego bohatera, *Winicjusz*. Duża liczba różnych form (22,5 tys. przy 135 tys. tokenów) wynika z bogatej fleksji polszczyzny. To argument za modelem znakowym: słownik słów byłby ogromny, a słownik znaków ma tylko 83 pozycje.

## 5. Wnioski

1. Wybrany korpus (*Quo vadis*, ok. 1,09 mln znaków po oczyszczeniu) jest wystarczająco duży i jednolity stylistycznie do treningu Char-RNN. Czyszczenie usunęło nagłówek, stopkę i znaki `\r` bez naruszenia treści utworu.
2. Podział 80/10/10 na ciągłe fragmenty, wykonany przed tworzeniem okien, eliminuje przeciek danych między zbiorami. Każda część zawiera ponad 100 tys. par.
3. Słownik zbudowany na treningu (`V = 83`) pokrywa walidację w całości, a test z wyjątkiem 3 wystąpień litery `Q`, obsłużonych przez `UNK`. Kodowanie i dekodowanie odtwarza tekst bezbłędnie.
4. Batche ze wszystkich zbiorów mają poprawny kształt, typ i zakres ID, a sprawdzone pary wejście–cel zgadzają się z tekstem źródłowym.
5. Długość okna praktycznie nie wpływa na liczbę par (spadek o `SEKW` par na fragment), ale zwiększa kontekst i koszt obliczeń epoki proporcjonalnie do `SEKW`. Wartość `SEKW = 40` jest rozsądnym punktem startowym.
6. Kierunki dalszej pracy: na laboratorium 7 trzeba zmierzyć czas epoki i stratę walidacyjną dla kilku `SEKW`. Przy długim czasie epoki można zwiększyć `KROK` lub losować podzbiór okien. Warto też rozważyć ujednolicenie rzadkich znaków (np. `>`, `«»`).

Dane (podział w plikach `quovadis_*.txt` i słownik w `slownik.json`) zostały zapisane i będą użyte bez zmian na kolejnych zajęciach.

---

Prowadzący: dr Tomasz Cieplak / mgr inż. Patryk Marek, Rok akademicki: 2026/27
