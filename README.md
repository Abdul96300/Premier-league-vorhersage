# Premier-League-Spielvorhersage

Eine Python-Webapp, die Ergebnisse von Premier-League-Spielen mit einem
**Poisson-Modell** vorhersagt. Man wählt Heim- und Auswärtsteam, und die App
berechnet die Wahrscheinlichkeiten für Sieg, Unentschieden und Niederlage
sowie die wahrscheinlichsten Endergebnisse.

Es ist mein erstes eigenständiges Python-Projekt. Mir ging es vor allem
darum, jeden Teil selbst zu verstehen, statt eine fertige Bibliothek
aufzurufen.

## Funktionsweise

Aus den Toren einer ganzen Saison schätzt das Modell für jedes Team eine
**Angriffsstärke** und eine **Abwehrstärke**, jeweils relativ zum
Liga-Durchschnitt. Daraus ergeben sich die erwarteten Tore eines Spiels. Die
Poisson-Verteilung rechnet diese erwarteten Tore dann in eine
Wahrscheinlichkeit für jedes mögliche Ergebnis um.

Ich habe **bewusst kein scikit-learn** oder ML-Framework benutzt. Tore sind
Zähldaten, und die Poisson-Verteilung ist genau dafür gemacht. Außerdem
kann ich so jede Zahl bis zur Formel zurückverfolgen und erklären.

## Tech-Stack

- **Python 3**
- **Flask** für Server und Routing
- **pandas** und **numpy** für die Daten
- **Jinja-Templates** mit handgeschriebenem CSS, ohne Tailwind oder Bootstrap

Die Logik in `model.py` ist von der Web-Schicht in `app.py` getrennt. Das
Modell weiß nichts von Flask und lässt sich allein testen.

## Daten

Saison 2023/24, als CSV von [football-data.co.uk](https://www.football-data.co.uk/).
Die Daten sind öffentlich, ein API-Key ist nicht nötig.

## Lokal starten

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Dann im Browser öffnen: http://localhost:5000

## Bekannte Eigenschaft

Die Wahrscheinlichkeiten summieren sich auf etwa 99 bis 100 Prozent, nicht
auf exakt 100. Sehr hohe Ergebnisse mit mehr als 10 Toren pro Team werden
abgeschnitten. Bei `max_tore=10` ist die Abweichung vernachlässigbar.


## Wie gut ist das Modell? (Backtest)

Um das Modell ehrlich zu prüfen, berechne ich die Teamstärken nur aus der
Hinrunde 2023/24 (188 Spiele bis 28.12.2023) und sage damit die 192 Spiele
der Rückrunde vorher. Zum Vergleich dienen eine Basislinie, die für jedes
Spiel nur die Ergebnis-Häufigkeiten der Hinrunde ansetzt, und die Quoten des
Buchmachers Pinnacle (ohne Marge).

| Modell | Accuracy | Log-Loss | Brier-Score |
|---|---:|---:|---:|
| Basislinie | 45,8 % | 1,070 | 0,646 |
| Poisson-Modell | 56,2 % | 0,917 | 0,540 |
| Buchmacher (Pinnacle) | 61,5 % | 0,879 | 0,512 | 
| Poisson-Modell (Walk-Forward) | 56,8 % | 0,914 | 0,537 |

Zusätzlich gibt es eine Walk-Forward-Variante: Vor jedem Spieltag werden die
Stärken aus allen bis dahin gespielten Spielen neu berechnet, so wie man es in
der Praxis tun würde. Das verbessert alle drei Kennzahlen nur minimal, der
wahrscheinlichste Ausgang ändert sich bei 11 von 192 Spielen. Der Abstand zum
Buchmacher liegt also kaum an veralteten Stärken, sondern eher daran, dass das
Modell nur Tore kennt und alle Spiele gleich stark gewichtet.

Das Poisson-Modell schlägt die Basislinie deutlich, liegt aber messbar hinter
dem Buchmacher. Das ist erwartbar: Pinnacle berücksichtigt Verletzungen,
Aufstellungen und aktuelle Form, mein Modell nur die Tore der Hinrunde.

Einschränkungen: nur eine Saison und 192 Testspiele, die Stärken werden
während der Rückrunde nicht aktualisiert, und keins der Modelle sagt je ein
Unentschieden als wahrscheinlichsten Ausgang voraus.

## Backtest über mehrere Saisons

Mit `backtest_saisons.py` sage ich drei Saisons vorher: 2023/24, 2024/25
und 2025/26. Vor jedem Spieltag werden die Teamstärken aus allen Spielen strikt
vor diesem Datum neu berechnet (Walk-Forward). Dabei vergleiche ich zwei
Trainingsfenster:

- **Fenster A:** nur laufende und vorige Saison
- **Fenster B:** alle Spiele ab 2021/22

Zum Vergleich dienen eine Basislinie (Ergebnis-Häufigkeiten aller bisherigen
Spiele) und die Quoten des Buchmachers Pinnacle (ohne Marge).

**Daten:** Premier League 2021/22 bis 2025/26 von
[football-data.co.uk](https://www.football-data.co.uk/), geladen mit
`python3 lade_daten.py`. Die Corona-Saisons 2019/20 und 2020/21 fehlen bewusst,
weil ohne Zuschauer kaum Heimvorteil bestand. Für 2025/26 liefert die Quelle
ab dem 17.01.2026 keine Pinnacle-Quoten. Diese 170 Spiele sind für alle
Modelle aus der Bewertung genommen, 2025/26 zählt also nur bis Anfang Januar.

**Aufsteiger:** Ein erster Lauf zeigte einen Fehler: Nach nur einem Spiel ohne
Gegentor bekam ein Aufsteiger die Abwehrstärke 0, und das Modell gab dem
Gegner 0 % Siegchance. Deshalb mische ich die Stärke von Aufsteigern mit dem
Durchschnitt der Absteiger der Vorsaison:
`(n · eigene Stärke + 10 · Ersatzwert) / (n + 10)`, mit n = eigene Spiele im
Training. Den Wert 10 habe ich vor der Auswertung festgelegt und danach nicht
mehr angepasst.

| Modell (970 Spiele) | Accuracy | Log-Loss | Brier-Score |
|---|---:|---:|---:|
| Basislinie | 43,8 % | 1,069 | 0,647 |
| Poisson, Fenster A | 53,2 % | 0,972 | 0,579 |
| Poisson, Fenster B | 53,0 % | 0,972 | 0,579 |
| Buchmacher (Pinnacle) | 55,6 % | 0,950 | 0,563 |

Log-Loss pro Saison:

| Saison | Spiele | Basislinie | Fenster A | Fenster B | Pinnacle |
|---|---:|---:|---:|---:|---:|
| 2023/24 | 380 | 1,055 | 0,942 | 0,926 | 0,909 |
| 2024/25 | 380 | 1,082 | 0,975 | 1,002 | 0,970 |
| 2025/26 | 210 | 1,072 | 1,022 | 1,001 | 0,989 |

**Ergebnis:** Beide Fenster schlagen die Basislinie in jeder Saison. Mehr
Geschichte (B) hilft im Mittel weder, noch schadet sie: Der Unterschied im
Log-Loss beträgt −0,0003 ± 0,0112, und der bessere Wert wechselt von Saison zu
Saison. Hinter Pinnacle liegt das Modell dagegen messbar zurück: +0,0218 ±
0,0132 im Log-Loss (Mittelwert der Differenzen pro Spiel ± 1,96
Standardfehler, eine grobe Faustregel und kein formaler Test). In keinem
Testspiel gibt ein Modell dem echten Ausgang weniger als 1 %.

Reproduzieren: `python3 lade_daten.py` und danach `python3 backtest_saisons.py`
(Ergebnis in `results/backtest_saisons.md`).

Reproduzieren: `python3 evaluate.py` (Ergebnis in `results/backtest.md`).