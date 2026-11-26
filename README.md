# MNIST Classifier

Jednoduchá neurónová sieť v PyTorch na rozpoznávanie ručne písaných číslic
0–9 z datasetu MNIST. Vstupom sú obrázky s rozmermi 28 × 28 pixelov.

## Inštalácia a spustenie

Projekt vyžaduje Python 3.10 alebo novší a verziu Pythonu podporovanú
nainštalovanými balíkmi PyTorch a torchvision.

V termináli otvor koreňový priečinok projektu (ten, ktorý obsahuje `main.py`
a `requirements.txt`). Všetky nasledujúce príkazy spúšťaj v tomto priečinku.

### Linux

Pri prvom použití vytvor virtuálne prostredie a nainštaluj závislosti:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Pred tréningom vytvor priečinok na uloženie modelu a spusti program:

```bash
mkdir -p model
python main.py
```

Ak si vo vetve `gui` a už máš natrénovaný model, grafické rozhranie spustíš:

```bash
python gui.py
```

Pri ďalšom otvorení terminálu stačí aktivovať prostredie pomocou
`source .venv/bin/activate` a spustiť `python main.py` alebo `python gui.py`.

### Windows (PowerShell)

Pri prvom použití vytvor virtuálne prostredie a nainštaluj závislosti:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pred tréningom vytvor priečinok na uloženie modelu a spusti program:

```powershell
mkdir model
python main.py
```

Ak si vo vetve `gui` a už máš natrénovaný model, grafické rozhranie spustíš:

```powershell
python gui.py
```

Pri ďalšom otvorení PowerShellu stačí aktivovať prostredie pomocou
`.\.venv\Scripts\Activate.ps1` a spustiť `python main.py` alebo `python gui.py`.
Ak Windows nerozpozná príkaz `python`, na vytvorenie prostredia skús
`py -m venv .venv`. Po aktivácii používaj `python`.

Súbor `requirements.txt` obsahuje `torch` a `torchvision` na tréning
a načítanie datasetu a `matplotlib` na vykresľovanie obrázkov v pomocných
skriptoch.

## Dáta a ukladanie modelu

- `data/` nemusíš vytvárať ručne. Pri prvom spustení `python main.py` ho
  knižnica torchvision vytvorí a stiahne doň MNIST. Na prvé stiahnutie je
  potrebný internet; pri ďalších spusteniach sa použijú existujúce dáta.
- `model/` musíš v aktuálnej verzii vytvoriť pred tréningom príkazom uvedeným
  vyššie. Program tento priečinok automaticky nevytvára a bez neho uloženie
  po tréningu zlyhá.
- Súbor `model/mnist_model.pth` nemusíš vytvárať ručne. Po tréningu ho
  `torch.save(model.state_dict(), ...)` vytvorí a uloží doň naučené váhy modelu.
  Architektúra siete zostáva definovaná v `model.py`.

Program `main.py` natrénuje model, vypíše priemernú stratu po každej epoche
a presnosť na testovacích dátach. Potom uloží váhy do `model/mnist_model.pth`
a načíta ich do novej inštancie modelu v režime vyhodnocovania.

Každé spustenie `python main.py` trénuje nový model a prepíše uložené váhy;
nepokračuje v predchádzajúcom tréningu. Cesty `data/` a `model/` sa pri tréningu
vyhodnocujú voči aktuálnemu priečinku terminálu, preto ho spúšťaj z koreňa projektu.
Aktuálny kód počíta na CPU.

`python gui.py` potrebuje už existujúci súbor `model/mnist_model.pth`, ale
nepotrebuje priečinok `data/`. Ak váhy ešte nemáš, najprv vytvor `model/`
a spusti `python main.py`.

## Grafické rozhranie (Tkinter)

Vo vetve `gui` po aktivácii virtuálneho prostredia podľa svojho systému spusti
v termináli v priečinku projektu:

```bash
python gui.py
```

Nakresli jednu číslicu ľavým tlačidlom myši a stlač **Rozpoznať** alebo Enter.
Tlačidlo **Vymazať** alebo Esc vyčistí plátno. Vedľa kresby uvidíš výsledok
a skóre všetkých číslic; dole je náhľad skutočného vstupu modelu.

GUI načítava `model/mnist_model.pth` vzhľadom na umiestnenie `gui.py`, takže
funguje aj pri spustení z iného priečinka. Netrénuje model a neprepisuje váhy.
Kresbu oreže, zmenší, vycentruje do obrázka 28 × 28 a normalizuje rovnako
ako tréningové dáta. Kresli veľké, zreteľné číslice; výsledok nemusí byť vždy správny.

Vyžaduje Pillow (súčasť `requirements.txt`) a Tkinter. Ak Python hlási chýbajúci
modul `tkinter`, doinštaluj systémový balík pre svoju verziu Pythonu
(napríklad `python3-tk` na Ubuntu/Debiane alebo `python3-tkinter` na Fedore).
Okno potrebuje grafické prostredie.

## Model a tréning

- Architektúra: `Flatten → Linear(784, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 10)`.
- Optimalizátor: Adam s learning rate `0.001`.
- Stratová funkcia: `CrossEntropyLoss`.
- Predvolený počet epoch: `5`; veľkosť dávky: `64`.
- Obrázky sa prevedú na tenzory a normalizujú hodnotami `0.1307` a `0.3081`.
  Tréningové obrázky navyše používajú náhodné otočenie do ±10°.

Počet epoch zmeníš v `main.py`, napríklad na `train(model, train_loader, epochs=10)`.
Veľkosť dávky a learning rate sú nastavené v `train.py`.

## Súbory projektu

| Súbor              | Úloha                                                       |
| ------------------ | ----------------------------------------------------------- |
| `main.py`          | Spustenie tréningu, vyhodnotenie, uloženie a načítanie váh. |
| `model.py`         | Definícia triedy `MNISTClassifier`.                         |
| `gui.py`           | Tkinter plátno a rozpoznávanie číslic uloženým modelom.     |
| `train.py`         | Načítanie dát, transformácie a tréningová slučka.           |
| `evaluate.py`      | Výpočet presnosti na testovacej množine.                    |
| `requirements.txt` | Závislosti projektu na inštaláciu cez pip.                  |
| `dataset.py`       | Rozpracovaný pomocný skript na načítanie dát.               |
| `main_notes.py`    | Rozpracované poznámky a vizualizácia obrázkov.              |

`main_notes.py` zatiaľ nie je spustiteľný: importuje `load_datasets`, ktorá
v `dataset.py` nie je definovaná. Hlavný program tieto pomocné súbory nepoužíva.

Dataset, uložené modely, virtuálne prostredia a generované Python súbory sú
ignorované cez `.gitignore`; zdrojové skripty zostávajú verzovateľné.
