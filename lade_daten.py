# lade_daten.py
# Laedt die Premier-League-Ergebnisse mehrerer Saisons von football-data.co.uk
# und speichert sie als data/saisons/E0_<SAISON>.csv.
#
# Warum ein eigenes Skript? So ist nachvollziehbar und wiederholbar, woher die
# Daten kommen. Wer das Projekt neu aufsetzt, muss nur dieses Skript starten,
# statt die Dateien von Hand herunterzuladen.
#
# Start im Projektordner:  python3 lade_daten.py

import os
import urllib.request

# Saisons im Kurzformat von football-data.co.uk: "2122" = Saison 2021/22.
#
# Bewusst NICHT dabei: 2019/20 und 2020/21. In diesen Corona-Saisons wurde
# (ganz oder groesstenteils) ohne Zuschauer gespielt, und der Heimvorteil war
# fast verschwunden. Das Modell lernt den Heimvorteil aber aus den Daten
# (liga_heim_schnitt gegen liga_auswaerts_schnitt). Diese Saisons wuerden ihn
# fuer normale Spiele mit Publikum falsch einschaetzen.
SAISONS = ["2122", "2223", "2324", "2425", "2526"]

# {saison} wird spaeter durch den Saison-Code ersetzt, z.B. .../2324/E0.csv.
# E0 ist bei football-data.co.uk das Kuerzel fuer die Premier League.
URL_MUSTER = "https://www.football-data.co.uk/mmz4281/{saison}/E0.csv"
ZIEL_ORDNER = "data/saisons"


def lade_saison(saison):
    """
    Laedt eine Saison herunter, prueft grob den Inhalt und speichert sie.
    Gibt den Dateipfad zurueck.
    """
    url = URL_MUSTER.format(saison=saison)
    ziel = os.path.join(ZIEL_ORDNER, f"E0_{saison}.csv")

    # Datei herunterladen. read() liefert die rohen Bytes der Antwort.
    with urllib.request.urlopen(url, timeout=30) as antwort:
        inhalt = antwort.read()

    # Plausibilitaetscheck: Eine echte Datei von football-data.co.uk beginnt
    # mit der Spalte "Div" (Division). Kommt stattdessen z.B. eine HTML-
    # Fehlerseite zurueck, wollen wir das sofort merken - und nicht erst
    # spaeter im Backtest mit einer unverstaendlichen Fehlermeldung.
    # Manche Dateien beginnen mit einem unsichtbaren Zeichen (BOM, Byte Order
    # Mark), deshalb wird das vor dem Vergleich entfernt.
    anfang = inhalt[:20].lstrip(b"\xef\xbb\xbf")
    if not anfang.startswith(b"Div"):
        raise ValueError(f"{url} sieht nicht wie eine football-data-CSV aus: {anfang!r}")

    # "wb" = als Bytes schreiben, also genau so speichern, wie die Datei ankam.
    with open(ziel, "wb") as datei:
        datei.write(inhalt)

    return ziel


if __name__ == "__main__":
    # Zielordner anlegen, falls er noch nicht existiert.
    os.makedirs(ZIEL_ORDNER, exist_ok=True)

    for saison in SAISONS:
        pfad = lade_saison(saison)
        groesse_kb = os.path.getsize(pfad) / 1024
        print(f"{saison}: gespeichert als {pfad} ({groesse_kb:.0f} KB)")
