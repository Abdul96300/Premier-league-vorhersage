# evaluate.py
# Backtest fuer das Poisson-Modell aus model.py.
#
# Die Frage: Wie gut sagt das Modell Spiele vorher, die es beim Berechnen der
# Staerken NICHT gesehen hat? Die App berechnet die Staerken aus der ganzen
# Saison. Wuerde man damit Spiele derselben Saison "vorhersagen", kennt das
# Modell die Ergebnisse schon - das waere wie eine Klausur, deren Loesungen man
# vorher gesehen hat. Deshalb trennen wir hier streng nach Zeit.
#
# Vorgehen:
#   1. Saison nach Datum sortieren und in der Mitte teilen.
#        Hinrunde   = Training (nur daraus werden die Staerken berechnet)
#        Rueckrunde = Test     (diese Spiele werden vorhergesagt)
#   2. Fuer jedes Testspiel drei Vorhersagen (Heimsieg / Remis / Auswaertssieg):
#        a) Basislinie: die Ergebnis-Haeufigkeiten der Hinrunde
#        b) das Poisson-Modell aus model.py, nur auf der Hinrunde berechnet
#        c) der Buchmacher Pinnacle als Vergleichsmassstab
#   3. Alle drei mit denselben drei Kennzahlen bewerten.
#
# Diese Datei aendert nichts an model.py oder app.py, sie benutzt model.py nur.
# Start im Projektordner:  python3 evaluate.py

import os

import numpy as np
import pandas as pd

import model

DATEN_PFAD = "data/E0.csv"
ERGEBNIS_PFAD = "results/backtest.md"

# Die feste Reihenfolge der drei Ausgaenge. Sie gilt fuer ALLE Arrays in dieser
# Datei: Spalte 0 = Heimsieg (H), Spalte 1 = Remis (D, "draw"),
# Spalte 2 = Auswaertssieg (A). Die Buchstaben sind die aus der CSV-Spalte FTR.
AUSGAENGE = ["H", "D", "A"]


# ---------------------------------------------------------------------------
# 1. Daten laden und zeitlich teilen
# ---------------------------------------------------------------------------

def lade_saison_mit_datum(pfad):
    """
    Liest die komplette CSV ein und sortiert die Spiele nach Datum.

    Warum nicht model.lade_spiele()? Die Funktion wirft beim Einlesen alles
    ausser Teams und Toren weg. Fuer den Backtest brauchen wir aber zusaetzlich
    das Datum (fuer den Split), das Endergebnis FTR und die Quoten.
    """
    saison = pd.read_csv(pfad)

    # Das Datum steht als Text im Format TT/MM/JJJJ in der CSV. Als Text
    # sortiert waere "01/02/2024" vor "11/08/2023" - deshalb in ein echtes
    # Datum umwandeln. format=... legt fest, dass der Tag zuerst kommt.
    saison["datum"] = pd.to_datetime(saison["Date"], format="%d/%m/%Y")

    # Nach Datum sortieren. kind="stable" behaelt bei gleichem Datum die
    # Reihenfolge aus der Datei bei, damit das Ergebnis immer gleich ist.
    saison = saison.sort_values("datum", kind="stable")
    saison = saison.reset_index(drop=True)

    return saison


def teile_saison(saison):
    """
    Teilt die Saison zeitlich in Hinrunde (Training) und Rueckrunde (Test).

    Warum kein zufaelliger Split? In der Realitaet kennt man vor einem Spiel
    nur die Vergangenheit. Ein zufaelliger Split wuerde Spiele aus dem Mai ins
    Training mischen, um Spiele aus dem Januar vorherzusagen - das Modell
    wuerde also in die Zukunft schauen und zu gut aussehen.

    Warum am Datum schneiden und nicht genau bei Spiel Nummer 190? Die
    Saisonmitte faellt auf einen Spieltag, an dem mehrere Spiele stattfinden.
    Wir nehmen das Datum des mittleren Spiels als Stichtag: alles VOR diesem
    Tag ist Training, alles AB diesem Tag ist Test. So liegt jedes Testspiel
    zeitlich nach jedem Trainingsspiel. Die Haelften sind dadurch nicht exakt
    gleich gross, aber sauber getrennt.
    """
    mitte = len(saison) // 2
    stichtag = saison["datum"].iloc[mitte]

    hinrunde = saison[saison["datum"] < stichtag]
    rueckrunde = saison[saison["datum"] >= stichtag]

    return hinrunde, rueckrunde


def ins_modellformat(spiele):
    """
    Bringt Spiele in genau das Format, das model.berechne_staerken() erwartet:
    vier Spalten mit den deutschen Namen, wie sie auch lade_spiele() erzeugt.
    """
    spiele = spiele[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    spiele = spiele.rename(columns={
        "HomeTeam": "heim",
        "AwayTeam": "auswaerts",
        "FTHG": "heimtore",
        "FTAG": "auswaertstore",
    })
    return spiele


def tatsaechliche_ausgaenge(rueckrunde):
    """
    Wandelt die echten Ergebnisse der Testspiele in eine Tabelle aus 0 und 1
    um (sogenanntes One-Hot-Encoding). Jede Zeile ist ein Spiel, in genau einer
    der drei Spalten steht eine 1 - beim Ausgang, der wirklich passiert ist.

        Heimsieg      -> [1, 0, 0]
        Remis         -> [0, 1, 0]
        Auswaertssieg -> [0, 0, 1]

    Warum diese Form? Dann haben echte Ergebnisse und Vorhersagen dieselbe
    Form (Zeilen = Spiele, 3 Spalten), und die Kennzahlen lassen sich direkt
    Spalte fuer Spalte vergleichen.
    """
    anzahl_spiele = len(rueckrunde)
    wahr = np.zeros((anzahl_spiele, 3))

    for zeile, ergebnis in enumerate(rueckrunde["FTR"]):
        spalte = AUSGAENGE.index(ergebnis)   # "H" -> 0, "D" -> 1, "A" -> 2
        wahr[zeile, spalte] = 1.0

    return wahr


# ---------------------------------------------------------------------------
# 2. Die drei Vorhersage-Varianten
#    Jede gibt ein Array zurueck: eine Zeile pro Testspiel, drei Spalten
#    (P(Heimsieg), P(Remis), P(Auswaertssieg)), jede Zeile summiert sich zu 1.
# ---------------------------------------------------------------------------

def basislinie_vorhersagen(hinrunde, anzahl_testspiele):
    """
    a) Basislinie: Wie oft gab es in der Hinrunde Heimsieg, Remis und
       Auswaertssieg? Genau diese Anteile sagen wir fuer JEDES Testspiel
       voraus - egal wer spielt.

    Warum so ein "dummes" Modell? Es ist die Messlatte. Ein Modell, das die
    Teams kennt, muss mindestens besser sein als eines, das nur weiss, dass
    Heimteams oft gewinnen. Sonst bringt die ganze Staerke-Berechnung nichts.
    """
    anteile = []
    for ausgang in AUSGAENGE:
        # (hinrunde["FTR"] == ausgang) ist eine Spalte aus True/False.
        # mean() davon ist der Anteil der True-Werte, also die Haeufigkeit.
        anteile.append((hinrunde["FTR"] == ausgang).mean())

    # Dieselbe Zeile fuer jedes Testspiel wiederholen.
    return np.array([anteile] * anzahl_testspiele)


def poisson_vorhersagen(hinrunde, rueckrunde):
    """
    b) Das eigene Poisson-Modell. Die Staerken werden NUR aus der Hinrunde
       berechnet - das ist der entscheidende Punkt gegen ein Datenleck.
       Danach wird fuer jedes Testspiel model.vorhersage() aufgerufen,
       also exakt dieselbe Funktion, die auch die App benutzt.
    """
    modell_hinrunde = model.berechne_staerken(ins_modellformat(hinrunde))

    zeilen = []
    for heim, auswaerts in zip(rueckrunde["HomeTeam"], rueckrunde["AwayTeam"]):
        ergebnis = model.vorhersage(modell_hinrunde, heim, auswaerts)
        p = [ergebnis["p_heimsieg"], ergebnis["p_remis"], ergebnis["p_auswaertssieg"]]

        # Normalisieren: model.vorhersage() rechnet nur bis max_tore=10 Tore pro
        # Team, deshalb ist die Summe minimal kleiner als 1 (siehe README).
        # Damit alle drei Varianten fair verglichen werden, teilen wir durch
        # die Summe. An den Verhaeltnissen aendert das praktisch nichts.
        summe = sum(p)
        zeilen.append([wert / summe for wert in p])

    return np.array(zeilen)


def buchmacher_vorhersagen(rueckrunde):
    """
    c) Buchmacher Pinnacle. Die Spalten PSH, PSD, PSA sind die Quoten fuer
       Heimsieg, Remis und Auswaertssieg, erhoben VOR dem Spiel - also kein
       Datenleck, diese Information war vor dem Anpfiff oeffentlich.

    Von der Quote zur Wahrscheinlichkeit: Eine Quote von 2.0 bedeutet
    "fuer 1 Euro Einsatz gibt es 2 Euro zurueck", das entspricht 1/2.0 = 50 %.
    Addiert man 1/Quote fuer alle drei Ausgaenge, kommt aber etwas ueber 100 %
    heraus (z.B. 102 %). Dieser Ueberschuss ist die Marge des Buchmachers,
    sein eingebauter Gewinn. Teilt man jede Zeile durch ihre Summe, faellt die
    Marge heraus und es bleiben echte Wahrscheinlichkeiten, die zu 1 summieren.

    Warum Pinnacle? Pinnacle gilt als besonders scharfer Buchmacher mit kleiner
    Marge. Seine Quoten sind ein realistischer Massstab dafuer, was mit sehr
    viel mehr Information (Verletzungen, Aufstellungen, Form) moeglich ist.
    """
    quoten = rueckrunde[["PSH", "PSD", "PSA"]].to_numpy()
    roh = 1 / quoten

    # sum(axis=1) = Summe pro Zeile (pro Spiel). keepdims=True sorgt dafuer,
    # dass numpy jede Zeile durch IHRE eigene Summe teilt.
    return roh / roh.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 3. Kennzahlen (alle mit numpy selbst berechnet)
#    p    = Vorhersagen, Form (Anzahl Spiele, 3)
#    wahr = echte Ergebnisse als 0/1-Tabelle, gleiche Form
# ---------------------------------------------------------------------------

def accuracy(p, wahr):
    """
    Accuracy (Trefferquote): Wie oft war der wahrscheinlichste Ausgang
    auch der, der wirklich passiert ist?

        Accuracy = Anzahl richtiger Tipps / Anzahl Spiele

    argmax liefert pro Zeile die Spalte mit dem groessten Wert, also den
    "Tipp" des Modells (0 = Heimsieg, 1 = Remis, 2 = Auswaertssieg).

    Schwaeche: Es zaehlt nur richtig oder falsch. Ob das Modell 34 % oder 90 %
    fuer den Heimsieg gesagt hat, ist egal - Hauptsache, er war der groesste
    Wert. Wie sicher sich ein Modell war, geht verloren.
    """
    tipp = np.argmax(p, axis=1)
    ergebnis = np.argmax(wahr, axis=1)
    return np.mean(tipp == ergebnis)


def log_loss(p, wahr):
    """
    Log-Loss: Wie viel Wahrscheinlichkeit hat das Modell dem Ausgang gegeben,
    der wirklich passiert ist? Niedriger ist besser.

        Log-Loss = - Durchschnitt ueber alle Spiele von ln(p_wahrer_Ausgang)

    Beispiele fuer ein einzelnes Spiel:
        dem echten Ausgang 90 % gegeben -> -ln(0.90) = 0.11  (kleine Strafe)
        dem echten Ausgang 33 % gegeben -> -ln(0.33) = 1.11
        dem echten Ausgang  5 % gegeben -> -ln(0.05) = 3.00  (grosse Strafe)

    Warum der Logarithmus? Er bestraft ueberhebliche Fehler besonders hart:
    Wer sich sehr sicher ist und falsch liegt, verliert viel. So belohnt die
    Kennzahl Modelle, deren Prozentzahlen ehrlich sind.
    """
    # p * wahr laesst pro Zeile nur den Wert beim echten Ausgang uebrig
    # (die anderen werden mit 0 multipliziert), sum(axis=1) holt ihn heraus.
    p_wahrer_ausgang = np.sum(p * wahr, axis=1)

    # Schutz: ln(0) waere minus unendlich. Wir setzen eine winzige Untergrenze,
    # damit ein einzelner Wert von exakt 0 nicht die ganze Rechnung sprengt.
    p_wahrer_ausgang = np.clip(p_wahrer_ausgang, 1e-15, 1.0)

    return -np.mean(np.log(p_wahrer_ausgang))


def brier_score(p, wahr):
    """
    Brier-Score (Mehrklassen-Version): der quadrierte Abstand zwischen
    Vorhersage und echtem Ergebnis, ueber alle drei Ausgaenge. Niedriger ist
    besser, 0 waere perfekt, 2 das schlechtestmoegliche Ergebnis.

        Brier = Durchschnitt ueber alle Spiele von
                (p_H - wahr_H)^2 + (p_D - wahr_D)^2 + (p_A - wahr_A)^2

    Beispiel: Vorhersage [0.6, 0.25, 0.15], echt war Heimsieg [1, 0, 0]
        (0.6 - 1)^2 + (0.25 - 0)^2 + (0.15 - 0)^2 = 0.16 + 0.0625 + 0.0225 = 0.245

    Unterschied zum Log-Loss: Auch hier zaehlt, wie sicher das Modell war,
    aber Fehler werden durch das Quadrat milder bestraft als durch den
    Logarithmus. Ein einzelner grober Ausreisser faellt weniger ins Gewicht.
    """
    return np.mean(np.sum((p - wahr) ** 2, axis=1))


# ---------------------------------------------------------------------------
# 4. Ausgabe
# ---------------------------------------------------------------------------

def beschreibe_split(hinrunde, rueckrunde):
    """
    Erzeugt die Textzeilen, die den Split beschreiben. Wird fuer Konsole und
    Markdown-Datei gleichermassen benutzt, damit beide dasselbe zeigen.
    """
    zaehler = rueckrunde["FTR"].value_counts()
    return [
        f"Training (Hinrunde):  {len(hinrunde)} Spiele, "
        f"{hinrunde['datum'].min():%d.%m.%Y} bis {hinrunde['datum'].max():%d.%m.%Y}",
        f"Test (Rueckrunde):    {len(rueckrunde)} Spiele, "
        f"{rueckrunde['datum'].min():%d.%m.%Y} bis {rueckrunde['datum'].max():%d.%m.%Y}",
        f"Echte Ausgaenge im Test: Heimsieg {zaehler.get('H', 0)}, "
        f"Remis {zaehler.get('D', 0)}, Auswaertssieg {zaehler.get('A', 0)}",
    ]


def drucke_tabelle(split_zeilen, ergebnisse):
    """Gibt den Split und die Ergebnistabelle in der Konsole aus."""
    print("Backtest Saison 2023/24")
    print()
    for zeile in split_zeilen:
        print(zeile)
    print()

    print(f"{'Modell':<38}{'Accuracy':>10}{'Log-Loss':>11}{'Brier-Score':>13}")
    print("-" * 72)
    for name, acc, ll, bs in ergebnisse:
        print(f"{name:<38}{acc * 100:>8.1f} %{ll:>11.3f}{bs:>13.3f}")
    print()
    print("Accuracy: hoeher ist besser. Log-Loss und Brier-Score: niedriger ist besser.")


def schreibe_markdown(pfad, split_zeilen, ergebnisse):
    """Schreibt dieselben Ergebnisse als Markdown-Tabelle in eine Datei."""
    # Den Ordner results/ anlegen, falls es ihn noch nicht gibt.
    # exist_ok=True: kein Fehler, wenn er schon existiert.
    os.makedirs(os.path.dirname(pfad), exist_ok=True)

    zeilen = ["# Backtest Saison 2023/24", ""]
    for zeile in split_zeilen:
        # Zwei Leerzeichen am Zeilenende = Zeilenumbruch in Markdown.
        zeilen.append(zeile + "  ")
    zeilen.append("")
    zeilen.append("| Modell | Accuracy | Log-Loss | Brier-Score |")
    zeilen.append("|---|---:|---:|---:|")
    for name, acc, ll, bs in ergebnisse:
        zeilen.append(f"| {name} | {acc * 100:.1f} % | {ll:.3f} | {bs:.3f} |")
    zeilen.append("")
    zeilen.append("- **Accuracy:** Anteil der Spiele, bei denen der wahrscheinlichste "
                  "Ausgang eingetreten ist. Hoeher ist besser.")
    zeilen.append("- **Log-Loss:** Durchschnitt von -ln(Wahrscheinlichkeit des echten "
                  "Ausgangs). Bestraft sichere Fehlprognosen stark. Niedriger ist besser.")
    zeilen.append("- **Brier-Score:** Quadrierter Abstand zwischen Vorhersage und "
                  "Ergebnis ueber Heimsieg/Remis/Auswaertssieg (0 bis 2). Niedriger ist besser.")
    zeilen.append("")
    zeilen.append("Erzeugt mit `python3 evaluate.py`.")

    with open(pfad, "w", encoding="utf-8") as datei:
        datei.write("\n".join(zeilen) + "\n")


# ---------------------------------------------------------------------------
# Programmablauf
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    saison = lade_saison_mit_datum(DATEN_PFAD)
    hinrunde, rueckrunde = teile_saison(saison)

    # Sicherheitscheck gegen ein Datenleck: Das letzte Trainingsspiel muss vor
    # dem ersten Testspiel liegen. Wenn nicht, bricht das Programm hier ab,
    # statt stillschweigend geschoente Zahlen zu liefern.
    assert hinrunde["datum"].max() < rueckrunde["datum"].min(), "Datenleck im Split!"

    wahr = tatsaechliche_ausgaenge(rueckrunde)

    vorhersagen = {
        "Basislinie (Haeufigkeiten Hinrunde)": basislinie_vorhersagen(hinrunde, len(rueckrunde)),
        "Poisson-Modell (model.py)": poisson_vorhersagen(hinrunde, rueckrunde),
        "Buchmacher (Pinnacle)": buchmacher_vorhersagen(rueckrunde),
    }

    ergebnisse = []
    for name, p in vorhersagen.items():
        # Die Kennzahlen setzen voraus, dass jede Zeile eine vollstaendige
        # Wahrscheinlichkeitsverteilung ist (Summe 1). Lieber hier pruefen,
        # als mit falschen Eingaben scheinbar gueltige Zahlen zu berechnen.
        assert np.allclose(p.sum(axis=1), 1.0), f"{name}: Zeilen summieren nicht zu 1"

        ergebnisse.append((name, accuracy(p, wahr), log_loss(p, wahr), brier_score(p, wahr)))

    split_zeilen = beschreibe_split(hinrunde, rueckrunde)
    drucke_tabelle(split_zeilen, ergebnisse)
    schreibe_markdown(ERGEBNIS_PFAD, split_zeilen, ergebnisse)
    print()
    print(f"Gespeichert in {ERGEBNIS_PFAD}")
