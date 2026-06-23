# app.py
# Hier liegt der WEB-Teil: der Flask-Server mit seinen Routen.
# Die ganze Mathematik holen wir uns aus model.py - app.py rechnet selbst nichts.

from flask import Flask, render_template, request

import model

# 1. Die Flask-Anwendung erzeugen.
#    __name__ sagt Flask, wo es startet (in welcher Datei). Das braucht Flask
#    intern, um z.B. spaeter die templates/ und static/ Ordner zu finden.
app = Flask(__name__)

# 2. Beim Start einmalig die Daten laden und das Modell berechnen.
#    Das muss nur EINMAL passieren, nicht bei jeder Anfrage - die Liga-Staerken
#    aendern sich ja nicht. Darum steht es hier oben, ausserhalb der Routen.
spiele = model.lade_spiele("data/E0.csv")
modell = model.berechne_staerken(spiele)

# 3. Die Liste aller Teamnamen, alphabetisch sortiert.
#    Brauchen wir spaeter fuer die Dropdowns; jetzt zum Anzeigen zum Testen.
teams = sorted(modell["staerken"].keys())


# 4. Die erste Route: der Pfad "/" (die Startseite).
#    Der Decorator @app.route("/") sagt Flask: "Wenn jemand '/' aufruft,
#    fuehre die Funktion direkt darunter aus."
@app.route("/")
def startseite():
    # render_template laedt die Datei templates/index.html, setzt die uebergebenen
    # Werte ein (hier: teams) und gibt das fertige HTML an den Browser zurueck.
    # "teams=teams" heisst: im Template heisst die Variable "teams" und bekommt
    # den Wert unserer Python-Liste teams.
    return render_template("index.html", teams=teams)


# 5. Zweite Route: /vorhersage. methods=["POST"] heisst, sie reagiert nur auf
#    abgeschickte Formulardaten (POST), nicht auf normales Seiten-Aufrufen (GET).
@app.route("/vorhersage", methods=["POST"])
def zeige_vorhersage():
    # Die zwei gewaehlten Teams aus dem Formular lesen. Die Schluessel "heim"
    # und "auswaerts" sind genau die name="..."-Attribute der <select>-Felder.
    heim = request.form["heim"]
    auswaerts = request.form["auswaerts"]

    # Validierung: Ein Team kann nicht gegen sich selbst spielen.
    # Ist das der Fall, rechnen wir NICHT, sondern zeigen wieder die Startseite
    # mit einer Fehlermeldung. So bekommt der Nutzer freundlich Bescheid.
    if heim == auswaerts:
        return render_template(
            "index.html",
            teams=teams,
            fehler="Bitte zwei verschiedene Teams auswaehlen.",
        )

    # Die eigentliche Rechnung macht model.py - app.py reicht nur durch.
    ergebnis = model.vorhersage(modell, heim, auswaerts)

    # Ergebnis-Template laden und alle Werte uebergeben, die es anzeigen soll.
    return render_template(
        "ergebnis.html",
        heim=heim,
        auswaerts=auswaerts,
        ergebnis=ergebnis,
    )


# 6. Dieser Block startet den Server, wenn man "python3 app.py" aufruft.
#    debug=True bedeutet: bei Code-Aenderungen startet der Server automatisch
#    neu und zeigt im Browser hilfreiche Fehlermeldungen. Nur fuer Entwicklung.
if __name__ == "__main__":
    app.run(debug=True)
