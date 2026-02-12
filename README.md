# LED-Grid-Simulation – LED-Bestrahlungsstärke-Berechnung

## Projekt-Ziel
Dieses Python-Tool dient der Berechnung und Visualisierung der Bestrahlungsstärke mehrerer LEDs auf einer Fläche in definiertem Abstand.

---
```bash
## Projektstruktur
LED-Grid-Simulation/
│
├── main.py # Hauptskript für die Simulation
├── visualization.py # Visualisierungsfunktionen (Heatmaps)
├── requirements.txt # Python-Abhängigkeiten
├── README.md # Diese Dokumentation
├── input_data/ # Eingabedateien (Excel)
│ └── *.xlsx
└── LED-Grid-Simulation.exe # Optionale Windows-Executable

**Hinweis:** Temporäre Build-Ordner (`build/`, `dist/`, `__pycache__/`) sowie die Datei `main.spec` sind technische Artefakte und nicht Bestandteil der wissenschaftlich relevanten Projektstruktur.
```
---

## Dateien & Funktionen

| Datei / Ordner             | Funktion |
|----------------------------|----------|
| **main.py**                | Hauptskript zur Durchführung der Simulation. Steuert Datenimport, Berechnung der Bestrahlungsstärke und Ausgabe der Ergebnisse. |
| **visualization.py**       | Enthält Funktionen zur grafischen Darstellung (Heatmaps) der simulierten Bestrahlung. |
| **input_data/**            | Excel-Dateien mit LED-Parametern und Konfigurationen der Simulation. |
| **requirements.txt**       | Enthält alle Python-Abhängigkeiten für die Reproduzierbarkeit. |
| **README.md**              | Diese Dokumentation. |
| **LED-Grid-Simulation.exe**| Kompilierte Windows-Version zur direkten Ausführung ohne Python. |

---

## Systemvoraussetzungen

- Python ≥ 3.9  
- Pip installiert  

Optional: Windows-Betriebssystem für die EXE-Version.

---

## Installation

1. Wechsle in das Projektverzeichnis.
2. Installiere die Abhängigkeiten:

```bash
pip install -r requirements.txt

## Python-Ausführung
1. python main.py
2. Warten bis Eingabefenster geöffnet wird
3. Auswahl der Eingabe-Datei

## EXE-Ausführung
1. Doppelklick auf `main.exe` im Ordner `dist`
2. Warten bis Eingabefenster geöffnet wird

3. Auswahl der Eingabe-Datei
