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
