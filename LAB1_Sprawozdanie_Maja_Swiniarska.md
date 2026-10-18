**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy

**Zajęcia nr / temat:** 1. Sekwencje i sygnał mowy

**Autor / grupa:** Maja Świniarska GL.04

**Data:** 15.10.2026

## 1. Cel ćwiczenia

Celem ćwiczenia było zapoznanie się z podstawowymi metodami przetwarzania cyfrowego sygnału mowy. 

Zadanie polegało na wczytaniu trzech różnego rodzaju nagrań audio (samogłoska, wyraz, całe zdanie) o częstotliwości próbkowania 16 kHz (mono), a następnie wygenerowaniu i analizie ich reprezentacji: przebiegu czasowego, spektrogramu melowego (log-mel) oraz współczynników melcepstralnych (MFCC).

Dodatkowo zbadano wpływ parametrów okna (N_FFT oraz HOP_LENGTH) na rozdzielczość czasową i częstotliwościową spektrogramu.

## 2. Metoda i implementacja

* **Wykorzystane narzędzia:** Biblioteki `librosa` (do przetwarzania sygnału audio i ekstrakcji cech), `numpy` (do operacji numerycznych) i `matplotlib` (do wizualizacji wyników).
* **Konfiguracja:** Zdefiniowano stałe parametry bazowe:
  * Częstotliwość próbkowania (`SR`) = 16000 Hz
  * Długość okna FFT (`N_FFT`) = 512
  * Przesunięcie ramki (`HOP_LENGTH`) = 160 (odpowiada 10 ms)
  * Liczba pasm mel (`N_MELS`) = 64
* **Implementacja:** Funkcje `log_mel()` oraz `mfcc()` wykorzystujące odpowiednie metody z biblioteki `librosa`.
  Następnie w pętli przetworzono pliki `samogłoska.wav`, `wyraz.wav` i `zdanie.wav`, wyświetlając dla każdego z nich przebieg fali, spektrogram i wykres MFCC.
  Przetestowano wpływ parametrów na pliku `wyraz.wav`, generując dwa spektrogramy: jeden z priorytetem rozdzielczości czasowej (`n_fft=256`, `hop_length=80`), drugi częstotliwościowej (`n_fft=1024`, `hop_length=320`).
* **[Notatnik z kodem](https://github.com/majax23/Uczenie_glebokie_w_systemach_rozpoznawania_i_przetwarzania_mowy_MS/blob/main/LAB1_Zadanie_Maja_Swiniarska_mowa.ipynb)**


## 3. Wyniki

Wygenerowano wykresy dla trzech rodzajów nagrań:
  * Plik `samogloska.wav`
    <img width="989" height="889" alt="image" src="https://github.com/user-attachments/assets/dc8075f1-a8f0-43e5-bfb3-154b05b8dc5f" />

  * Plik `wyraz.wav`
    <img width="989" height="889" alt="image" src="https://github.com/user-attachments/assets/256b6dcf-0485-4cca-a7a2-ce43e274fdce" />

  * Plik `zdanie.wav`
    <img width="989" height="889" alt="image" src="https://github.com/user-attachments/assets/0fda7768-5321-4697-b797-c21069f9d539" />

* Wygenerowano wykres porównawczy ilustrujący wpływ parametrów okna na spektrogram dla pliku `wyraz.wav`.
  <img width="924" height="790" alt="image" src="https://github.com/user-attachments/assets/5d5fcad8-05ff-4625-ad01-704a6fc4ebd9" />


## 4. Analiza i interpretacja

### Porównanie nagrań
Zgodnie z przewidywaniami, charakterystyka reprezentacji silnie zależy od złożoności sygnału:
* **Samogłoska**: Przebieg czasowy po początkowym ataku jest bardzo jednorodny. Spektrogram log-mel i wykres MFCC wykazują stałe poziome pasma energii, co świadczy o stabilności widmowej izolowanej głoski.
* **Wyraz**: Sygnał jest bardziej złożony. W przebiegu czasowym widoczne są zmiany amplitudy. Na spektrogramie można zaobserwować dynamiczne zmiany struktury widmowej, odpowiadające przejściom między kolejnymi głoskami, w tym nagłe pionowe skoki energii oznaczające spółgłoski oraz poziome pasma samogłosek.
* **Zdanie**: Sygnał jest najbardziej zróżnicowany. Widoczne są wyraźne pauzy między wyrazami oraz dynamiczne, ciągłe zmiany rozkładu energii na spektrogramie i w MFCC.

### Wpływ parametrów (`N_FFT` i `HOP_LENGTH`)
Porównanie dwóch wariantów spektrogramu dla pliku `wyraz.wav` ukazuje klasyczny kompromis wynikający z transformaty Fouriera:
* **Małe okno** (`N_FFT=256`, `HOP_LENGTH=80`): Zapewnia wyższą rozdzielczość czasową. Krótkotrwałe zdarzenia, takie jak uderzenia spółgłosek zwartych, są bardzo dobrze zarysowane w osi czasu, jednak pasma częstotliwościowe (formanty) stają się szersze i rozmyte.
* **Duże okno** (`N_FFT=1024`, `HOP_LENGTH=320`): Zapewnia precyzyjniejszą rozdzielczość częstotliwościową. Poziome prążki energii są cienkie i wyraźnie odseparowane, jednakże tracimy dokładność w osi czasu (nagłe zdarzenia ulegają rozmyciu).

## 5. Wnioski

Wybór rodzaju reprezentacji i jej parametrów jest kluczowy w analizie mowy i zależy od konkretnego zadania. Do precyzyjnej lokalizacji w czasie lepsze będą krótsze okna FFT. Do analizy wysokości dźwięku lub struktury formantowej korzystniejsze będą dłuższe okna.

Reprezentacje log-mel i MFCC bardzo dobrze sprawdzają się w kompresowaniu informacji o sygnale audio, zachowując najważniejsze cechy percepcyjne mowy, przez co są standardem wejściowym dla sieci neuronowych w systemach ASR.

Mowa ciągła charakteryzuje się znacznie większą dynamiką zmian w czasie i częstotliwości w porównaniu do stabilnej, izolowanej samogłoski.

---
