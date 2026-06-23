# Session Hand-off: Premier-League-Spielvorhersager

## WICHTIG — Arbeitsweise (bitte einhalten)
Ich bin Wirtschaftsinformatik-Student im 2. Semester. Das ist ein LERNPROJEKT.
Ich will den Code selbst verstehen und im Bewerbungsgespraech erklaeren koennen.
Geschwindigkeit egal, Verstaendnis wichtig. Halte dich an:
- In kleinen Schritten bauen, nicht alles auf einmal.
- VOR jedem Schritt auf Deutsch in 2-3 Saetzen erklaeren, WAS und WARUM.
- NACH dem Schritt die wichtigsten Zeilen erklaeren.
- Einfacher, lesbarer Code. Keine cleveren Einzeiler, keine vorzeitigen
  Abstraktionen, keine unnoetigen Design Patterns. Deutsche Kommentare.
- Nach jedem groesseren Schritt PAUSIEREN und fragen, ob ich es verstanden habe,
  bevor es weitergeht.
- Bei zwei Wegen: den einfacheren, verstaendlicheren nehmen.

## 1. Projekt-Status — FUNKTIONSFAEHIG & KOMPLETT
Python-Webapp, die mit einem POISSON-MODELL Premier-League-Spiele vorhersagt.
Alle urspruenglich geplanten Schritte (1-7) sind erledigt. Die App laeuft
end-to-end: Formular -> POST -> Modell -> Ergebnis-Seite, inkl. Styling.

Architektur bewusst getrennt:
- model.py  = gesamte Vorhersage-Logik (Mathe), weiss NICHTS vom Web. FERTIG.
- app.py    = Flask-Routen (Web). FERTIG (2 Routen + Validierung).
Tech-Stack: Python 3, Flask, Jinja-Templates, handgeschriebenes CSS.
KEIN scikit-learn / ML-Framework. Nur pandas + numpy + math.
Daten: football-data.co.uk, Saison 2023/24, lokal als CSV.

Projektordner: /Users/abdul9630/premier-league-vorhersage
Struktur (alles vorhanden):
    premier-league-vorhersage/
    |-- data/E0.csv               (380 Spiele, Saison 2023/24)
    |-- templates/
    |   |-- index.html            (Formular: 2 Dropdowns + Fehlermeldung)
    |   |-- ergebnis.html         (Ergebnis-Tabellen)
    |-- static/css/style.css      (handgeschriebenes Design)
    |-- model.py                  (FERTIG - Poisson-Logik)
    |-- app.py                    (FERTIG - Flask-Server)
    |-- requirements.txt          (flask==3.1.2, pandas==2.3.3, numpy==2.3.3)
    |-- venv/                      (virtuelle Umgebung, lokal angelegt)
    |-- HANDOFF.md                (dieses Dokument)

## 2. Erledigt (komplette Historie)
Schritt 1: Ordnerstruktur angelegt, CSV per curl geladen
  (https://www.football-data.co.uk/mmz4281/2324/E0.csv -> data/E0.csv).
Schritt 2: model.py -> lade_spiele(pfad). Liest CSV mit pandas, reduziert auf
  4 Spalten, benennt sie deutsch um.
Schritt 3: model.py -> berechne_staerken(spiele). Liga-Durchschnitte +
  Angriffs-/Abwehrstaerke pro Team.
Schritt 4: model.py -> poisson_wahrscheinlichkeit, erwartete_tore, vorhersage.
  Komplettes Poisson-Modell, getestet (Man City vs Burnley ~90% Heimsieg).
Schritt 5: requirements.txt + app.py (Flask-Grundgeruest). venv angelegt,
  Pakete installiert, Server startet sauber. Erste Route "/" gab Teamliste aus.
Schritt 6a: templates/index.html (Formular, 2 Dropdowns, POST an /vorhersage).
  Route "/" auf render_template umgestellt.
  LERNPUNKT: Jinja parst die GANZE Datei nach {{ }}/{% %}, auch in HTML-
  Kommentaren -> keine echten Jinja-Klammern als Beispiel in Kommentare.
Schritt 6b: app.py -> Route /vorhersage (POST). Liest request.form["heim"] /
  ["auswaerts"], ruft model.vorhersage auf, rendert ergebnis.html.
  templates/ergebnis.html (erwartete Tore, 3 Wahrscheinlichkeiten, Top-4).
Schritt 6c (Zusatzwunsch): Validierung "nicht gegen sich selbst spielen".
  if heim == auswaerts -> index.html mit Fehlermeldung erneut anzeigen,
  KEINE Berechnung. Serverseitig (Begruendung: "never trust the client").
Schritt 7: static/css/style.css von Hand geschrieben. Design nuechtern,
  sportlich-redaktionell. Beide Templates verlinken CSS via url_for('static',...).
  border="1" aus HTML entfernt, Styling komplett ins CSS verlagert.

## 3. Aktueller Code-Stand (genaue Struktur)

model.py (unveraendert seit Schritt 4):
- import math; import pandas as pd
- lade_spiele(pfad) -> DataFrame: heim, auswaerts, heimtore, auswaertstore
- berechne_staerken(spiele) -> dict "modell" (siehe Abschnitt 5)
- poisson_wahrscheinlichkeit(k, lam) -> float
- erwartete_tore(modell, heim, auswaerts) -> (lam_heim, lam_auswaerts)
- vorhersage(modell, heim, auswaerts, max_tore=10) -> dict "ergebnis"
- if __name__ == "__main__": Testblock (Man City vs Burnley)

app.py:
- from flask import Flask, render_template, request; import model
- app = Flask(__name__)
- BEIM START einmalig: spiele = model.lade_spiele("data/E0.csv");
  modell = model.berechne_staerken(spiele);
  teams = sorted(modell["staerken"].keys())
- @app.route("/")                       -> render_template("index.html", teams=teams)
- @app.route("/vorhersage", POST)       -> liest 2 Teams, Validierung
  (heim==auswaerts -> index.html mit fehler=...), sonst model.vorhersage(...)
  -> render_template("ergebnis.html", heim, auswaerts, ergebnis)
- if __name__ == "__main__": app.run(debug=True)

## 4. Start-Anleitung (verifiziert)
    cd /Users/abdul9630/premier-league-vorhersage
    source venv/bin/activate          # venv aktivieren (zsh/macOS)
    pip install -r requirements.txt   # nur beim ersten Mal noetig
    python3 app.py                    # Server starten
    # Browser: http://localhost:5000/   (= http://127.0.0.1:5000)
    # Strg+C beendet den Server, "deactivate" verlaesst die venv.

## 5. Datenstrukturen (unveraendert, zur Referenz)
- DataFrame aus lade_spiele(): Spalten "heim", "auswaerts", "heimtore", "auswaertstore".
- "modell" (Rueckgabe berechne_staerken):
    {
      "liga_heim_schnitt": float,        # ~1.80
      "liga_auswaerts_schnitt": float,   # ~1.48
      "staerken": { "Arsenal": {"angriff": float, "abwehr": float}, ... }
    }
- "ergebnis" (Rueckgabe vorhersage):
    {
      "erwartete_heimtore": float,
      "erwartete_auswaertstore": float,
      "p_heimsieg": float,        # 0..1 (im Template *100 fuer Prozent)
      "p_remis": float,
      "p_auswaertssieg": float,
      "top_ergebnisse": [ (heimtore, auswaertstore, wahrscheinlichkeit), ... ]  # Top 4
    }
- Teamliste fuer Dropdowns: sorted(modell["staerken"].keys()).
  Teamnamen exakt wie in CSV (z.B. "Man City", "Nott'm Forest", "Sheffield United").

## 6. Modell-Logik (zum Erklaeren im Gespraech)
- Angriffsstaerke = Team-Tore-Schnitt / Liga-Schnitt-pro-Team (>1 = gut).
- Abwehrstaerke   = kassierte Tore-Schnitt / Liga-Schnitt-pro-Team (<1 = gut).
- erwartete Heimtore   = angriff(heim) * abwehr(auswaerts) * liga_heim_schnitt
- erwartete Auswaertst.= angriff(auswaerts) * abwehr(heim) * liga_auswaerts_schnitt
- P(Endstand i:j) = poisson(i, lam_heim) * poisson(j, lam_auswaerts) (Unabhaengigkeit).
- Raster 0..max_tore (=10) je Team; aufsummiert zu Sieg/Remis/Niederlage.

WICHTIGER BEKANNTER PUNKT (kein Bug, bewusste Eigenschaft):
- Wahrscheinlichkeiten summieren sich nur ~99-100%, weil Ergebnisse ueber
  max_tore abgeschnitten werden. Mit max_tore=10 vernachlaessigbar.

## 7. Umgebung
- Python 3.13.7, Flask 3.1.2, pandas 2.3.3, numpy 2.3.3 (in venv installiert).
- Plattform: macOS (darwin), Shell zsh. Python als "python3" aufrufen.
- Working directory: /Users/abdul9630/premier-league-vorhersage
- OFFENE BUGS: keine bekannt. Alle Routen mit Flask-Test-Client geprueft (Status 200).

## 8. Moegliche naechste Schritte (OPTIONAL, nicht begonnen)
Das Pflichtprojekt ist fertig. Falls weiter geuebt werden soll:
- Gewaehlte Teams auf der Ergebnis-Seite im Dropdown vorausgewaehlt lassen
  (z.B. ueber zurueckgegebene Werte + "selected" im <option>).
- README.md schreiben (Projektbeschreibung + Start-Anleitung) fuers Portfolio.
- Mehrere Saisons einlesen / kombinieren (war frueher eine offene Frage).
- Kleinen Automatik-Test (pytest) fuer model.vorhersage ergaenzen.
- Favicon hinzufuegen (entfernt die harmlosen 404 in den Server-Logs).

START-ANWEISUNG FUER NEUE SESSION:
"Lies HANDOFF.md im Ordner premier-league-vorhersage. Das Projekt ist fertig
und laeuft. Halte dich an die Arbeitsweise (erst erklaeren, kleiner Schritt,
erklaeren, pausieren). Frag mich, welchen der optionalen Schritte aus
Abschnitt 8 ich machen moechte - oder ob ich etwas am bestehenden Code
verstehen/aendern will."
