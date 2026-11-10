# Backtest Saison 2023/24

Training (Hinrunde):  188 Spiele, 11.08.2023 bis 28.12.2023  
Test (Rueckrunde):    192 Spiele, 30.12.2023 bis 19.05.2024  
Echte Ausgaenge im Test: Heimsieg 88, Remis 46, Auswaertssieg 58  

| Modell | Accuracy | Log-Loss | Brier-Score |
|---|---:|---:|---:|
| Basislinie (Haeufigkeiten Hinrunde) | 45.8 % | 1.070 | 0.646 |
| Poisson-Modell (model.py) | 56.2 % | 0.917 | 0.540 |
| Buchmacher (Pinnacle) | 61.5 % | 0.879 | 0.512 |

- **Accuracy:** Anteil der Spiele, bei denen der wahrscheinlichste Ausgang eingetreten ist. Hoeher ist besser.
- **Log-Loss:** Durchschnitt von -ln(Wahrscheinlichkeit des echten Ausgangs). Bestraft sichere Fehlprognosen stark. Niedriger ist besser.
- **Brier-Score:** Quadrierter Abstand zwischen Vorhersage und Ergebnis ueber Heimsieg/Remis/Auswaertssieg (0 bis 2). Niedriger ist besser.

Erzeugt mit `python3 evaluate.py`.
