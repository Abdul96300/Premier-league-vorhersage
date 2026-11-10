# Backtest ueber mehrere Saisons (Walk-Forward)

Daten: Premier League von football-data.co.uk, Saisons 2021/22, 2022/23, 2023/24, 2024/25, 2025/26. 2019/20 und 2020/21 bewusst ausgelassen (Corona, kaum Heimvorteil).

Vorhergesagt werden die Saisons 2023/24, 2024/25, 2025/26. Vor jedem Spieltag werden die Staerken aus allen Spielen strikt vor diesem Datum neu berechnet:

- **Fenster A (Aktuell + Vorsaison):** nur laufende und vorige Saison
- **Fenster B (Alle bisherigen):** alle Spiele ab 2021/22
- **Basislinie:** Ergebnis-Haeufigkeiten aller Spiele vor dem Datum
- **Pinnacle:** 1/Quote, normalisiert (ohne Marge)

Aufsteiger: Ersatzwert = Durchschnitt der drei Absteiger der Vorsaison. Gemischt mit der eigenen Staerke als (n * eigene + 10 * Ersatzwert) / (n + 10), n = eigene Spiele im Trainingsfenster. Ohne eigene Spiele gilt der reine Ersatzwert.

Spiele ohne Pinnacle-Quoten sind fuer alle Modelle aus der Bewertung genommen:

- 2023/24: 0 von 380 Spielen ausgelassen
- 2024/25: 0 von 380 Spielen ausgelassen
- 2025/26: 170 von 380 Spielen ausgelassen

| Saison | Modell | Spiele | Accuracy | Log-Loss | Brier-Score | P(echt) < 1 % |
|---|---|---:|---:|---:|---:|---:|
| 2023/24 | Basislinie (Haeufigkeiten) | 380 | 46.1 % | 1.055 | 0.637 | 0 |
| 2023/24 | Poisson A (Aktuell + Vorsaison) | 380 | 55.0 % | 0.942 | 0.556 | 0 |
| 2023/24 | Poisson B (Alle bisherigen) | 380 | 57.6 % | 0.926 | 0.546 | 0 |
| 2023/24 | Buchmacher (Pinnacle) | 380 | 58.9 % | 0.909 | 0.533 | 0 |
| 2024/25 | Basislinie (Haeufigkeiten) | 380 | 40.8 % | 1.082 | 0.656 | 0 |
| 2024/25 | Poisson A (Aktuell + Vorsaison) | 380 | 53.2 % | 0.975 | 0.583 | 0 |
| 2024/25 | Poisson B (Alle bisherigen) | 380 | 51.3 % | 1.002 | 0.600 | 0 |
| 2024/25 | Buchmacher (Pinnacle) | 380 | 53.9 % | 0.970 | 0.579 | 0 |
| 2025/26 | Basislinie (Haeufigkeiten) | 210 | 45.2 % | 1.072 | 0.648 | 0 |
| 2025/26 | Poisson A (Aktuell + Vorsaison) | 210 | 50.0 % | 1.022 | 0.613 | 0 |
| 2025/26 | Poisson B (Alle bisherigen) | 210 | 47.6 % | 1.001 | 0.598 | 0 |
| 2025/26 | Buchmacher (Pinnacle) | 210 | 52.4 % | 0.989 | 0.592 | 0 |
| Gesamt | Basislinie (Haeufigkeiten) | 970 | 43.8 % | 1.069 | 0.647 | 0 |
| Gesamt | Poisson A (Aktuell + Vorsaison) | 970 | 53.2 % | 0.972 | 0.579 | 0 |
| Gesamt | Poisson B (Alle bisherigen) | 970 | 53.0 % | 0.972 | 0.579 | 0 |
| Gesamt | Buchmacher (Pinnacle) | 970 | 55.6 % | 0.950 | 0.563 | 0 |

Accuracy: hoeher ist besser. Log-Loss und Brier-Score: niedriger ist besser. P(echt) < 1 %: Anzahl Spiele, bei denen das Modell dem echten Ausgang weniger als 1 % gab.

**Reiner Ersatzwert benutzt**, Team ganz ohne eigene Staerke (Anzahl Team-Einsaetze):

- 2023/24: Fenster A 5x, Fenster B 4x
- 2024/25: Fenster A 4x, Fenster B 1x
- 2025/26: Fenster A 4x, Fenster B 1x

**Log-Loss-Differenz pro Spiel**, alle Testsaisons, Mittelwert +/- 1,96 Standardfehler. Negativ = erstes Modell besser. Grobe Faustregel, kein formaler Test (Spiele sind nicht voellig unabhaengig voneinander).

- Poisson A minus Poisson B: -0.0003 +/- 0.0112 (nicht von Zufall zu unterscheiden)
- Poisson A (Aktuell + Vorsaison) minus Pinnacle: +0.0218 +/- 0.0132 (Bereich enthaelt die 0 nicht)

Erzeugt mit `python3 lade_daten.py` und `python3 backtest_saisons.py`.
