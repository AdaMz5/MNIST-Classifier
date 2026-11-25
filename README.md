# MNIST Classifier

Jednoduchá neurónová sieť v PyTorch na rozpoznávanie ručne písaných číslic
0–9 z datasetu MNIST. Vstupom sú obrázky s rozmermi 28 × 28 pixelov.

## Inštalácia

Projekt vyžaduje Python 3.10 alebo novší a verziu Pythonu podporovanú
nainštalovanými balíkmi PyTorch a torchvision.

V priečinku projektu vytvor a aktivuj virtuálne prostredie:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Vo Windows PowerShell namiesto `source` použi:

```powershell
.venv\Scripts\Activate.ps1
```

Súbor `requirements.txt` obsahuje `torch` a `torchvision` na tréning
a načítanie datasetu a `matplotlib` na vykresľovanie obrázkov v pomocných
skriptoch.

## Spustenie

Príkazy spúšťaj z koreňového priečinka projektu:

```bash
python -c "from pathlib import Path; Path('model').mkdir(exist_ok=True)"
python main.py
```

Pri prvom spustení sa MNIST stiahne do `data/`, preto je potrebné pripojenie
na internet. Program natrénuje model, vypíše priemernú stratu po každej epoche
a presnosť na testovacích dátach. Potom uloží váhy do `model/mnist_model.pth`
a načíta ich do novej inštancie modelu v režime vyhodnocovania.

Každé spustenie trénuje nový model a prepíše uložené váhy. Aktuálny kód
počíta na CPU.

## Grafické rozhranie (Tkinter)

Na rozpoznávanie vlastných číslic pomocou uloženého modelu spusti:

```bash
source .venv/bin/activate
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

| Súbor | Úloha |
| --- | --- |
| `main.py` | Spustenie tréningu, vyhodnotenie, uloženie a načítanie váh. |
| `model.py` | Definícia triedy `MNISTClassifier`. |
| `gui.py` | Tkinter plátno a rozpoznávanie číslic uloženým modelom. |
| `train.py` | Načítanie dát, transformácie a tréningová slučka. |
| `evaluate.py` | Výpočet presnosti na testovacej množine. |
| `requirements.txt` | Závislosti projektu na inštaláciu cez pip. |
| `dataset.py` | Rozpracovaný pomocný skript na načítanie dát. |
| `main_notes.py` | Rozpracované poznámky a vizualizácia obrázkov. |

`main_notes.py` zatiaľ nie je spustiteľný: importuje `load_datasets`, ktorá
v `dataset.py` nie je definovaná. Hlavný program tieto pomocné súbory nepoužíva.

Dataset, uložené modely, virtuálne prostredia a generované Python súbory sú
ignorované cez `.gitignore`; zdrojové skripty zostávajú verzovateľné.
