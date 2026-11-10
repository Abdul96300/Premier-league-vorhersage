# backtest_saisons.py
# Walk-Forward-Backtest des Poisson-Modells ueber mehrere Saisons.
#
# Warum zusaetzlich zu evaluate.py? Dort werden nur 192 Spiele einer Saison
# getestet. Bei so wenigen Spielen sind kleine Unterschiede zwischen Modellen
# nicht von Zufall zu unterscheiden. Hier werden drei komplette Saisons
# vorhergesagt (2023/24, 2024/25, 2025/26), mit Training aus bis zu fuenf
# Saisons (ab 2021/22).
#
# Vorgehen fuer jedes Datum einer Testsaison:
#   1. Staerken mit model.berechne_staerken() aus Spielen STRIKT VOR diesem
#      Datum berechnen - und zwar mit zwei verschiedenen Trainingsfenstern:
#        A) "Aktuell + Vorsaison": nur laufende und vorige Saison
#        B) "Alle bisherigen":     alle geladenen Spiele ab 2021/22
#   2. Aufsteiger: Ihre Staerke wird mit einem Ersatzwert gemischt, solange
#      sie im Training erst wenige Spiele haben (Shrinkage, siehe unten).
#   3. Die Spiele dieses Datums vorhersagen.
# Zum Vergleich: Basislinie (Ergebnis-Haeufigkeiten) und Pinnacle-Quoten.
#
# Daten: football-data.co.uk, geladen mit lade_daten.py nach data/saisons/.
# Diese Datei aendert nichts an evaluate.py, model.py oder app.py.
# Start im Projektordner:  python3 backtest_saisons.py

import os

import numpy as np
import pandas as pd

import model
from evaluate import (
    AUSGAENGE,
    accuracy,
    basislinie_vorhersagen,
    brier_score,
    buchmacher_vorhersagen,
    ins_modellformat,
    log_loss,
    poisson_fuer_spiele,
    tatsaechliche_ausgaenge,
)

DATEN_ORDNER = "data/saisons"
ERGEBNIS_PFAD = "results/backtest_saisons.md"

# Alle geladenen Saisons in zeitlicher Reihenfolge. Die Reihenfolge ist
# wichtig: Die "Vorsaison" einer Saison ist der Eintrag direkt davor.
# 2019/20 und 2020/21 fehlen bewusst (Corona, kaum Heimvorteil - siehe
# Begruendung in lade_daten.py).
SAISONS = ["2122", "2223", "2324", "2425", "2526"]

# Diese Saisons werden vorhergesagt. 2021/22 und 2022/23 dienen nur als
# Training, damit schon das erste Testspiel genug Vorgeschichte hat.
TESTSAISONS = ["2324", "2425", "2526"]

QUOTEN_SPALTEN = ["PSH", "PSD", "PSA"]

# Gewicht des Ersatzwerts beim Shrinkage fuer Aufsteiger (siehe
# ergaenze_aufsteiger). Der Ersatzwert zaehlt wie K zusaetzliche "virtuelle"
# Spiele: Nach 10 eigenen Spielen zaehlen eigene Daten und Ersatzwert je zur
# Haelfte. Der Wert wurde VOR dem Auswerten festgelegt und wird danach nicht
# mehr angepasst. Wuerde man K so lange veraendern, bis die Tabelle gut
# aussieht, haette man es unbemerkt an die Testdaten angepasst - genau die Art
# Datenleck, die der Backtest verhindern soll.
SHRINKAGE_K = 10

# Namen der Modelle, wie sie in den Tabellen stehen. Als Konstanten, damit
# Tippfehler sofort als Fehler auffallen statt als leere Tabellenzeile.
BASISLINIE = "Basislinie (Haeufigkeiten)"
POISSON_A = "Poisson A (Aktuell + Vorsaison)"
POISSON_B = "Poisson B (Alle bisherigen)"
PINNACLE = "Buchmacher (Pinnacle)"
MODELLE = [BASISLINIE, POISSON_A, POISSON_B, PINNACLE]


def saison_name(saison):
    """Macht aus dem Kurzformat "2324" die lesbare Form "2023/24"."""
    return f"20{saison[:2]}/{saison[2:]}"


def vorsaison(saison):
    """Gibt die Saison direkt vor der uebergebenen zurueck, z.B. "2324" -> "2223"."""
    position = SAISONS.index(saison)
    return SAISONS[position - 1]


# ---------------------------------------------------------------------------
# 1. Daten laden
# ---------------------------------------------------------------------------

def lies_datum(datumstexte):
    """
    Wandelt die Datumsspalte in echte Daten um und gibt zusaetzlich zurueck,
    welches Format erkannt wurde.

    Warum nicht einfach evaluate.lade_saison_mit_datum()? Die Funktion
    erwartet fest eine vierstellige Jahreszahl (TT/MM/JJJJ). Aeltere Dateien
    von football-data.co.uk benutzen teils eine zweistellige (TT/MM/JJ).
    Deshalb wird hier zuerst das vierstellige Format probiert und nur, wenn
    das scheitert, das zweistellige. Passt keins von beiden (oder sind die
    Formate innerhalb einer Datei gemischt), bricht das Programm mit einem
    Fehler ab, statt Daten falsch zu lesen.
    """
    try:
        return pd.to_datetime(datumstexte, format="%d/%m/%Y"), "TT/MM/JJJJ"
    except ValueError:
        return pd.to_datetime(datumstexte, format="%d/%m/%y"), "TT/MM/JJ"


def lade_alle_saisons():
    """
    Liest alle Saison-Dateien ein und haengt sie zu EINER Tabelle zusammen.
    Jedes Spiel bekommt eine Spalte "saison" (z.B. "2324") und "datum".

    Gibt die Tabelle und ein Dictionary mit dem erkannten Datumsformat pro
    Saison zurueck (fuer den Datencheck).
    """
    teile = []
    formate = {}

    for saison in SAISONS:
        spiele = pd.read_csv(os.path.join(DATEN_ORDNER, f"E0_{saison}.csv"))
        spiele["datum"], formate[saison] = lies_datum(spiele["Date"])
        spiele["saison"] = saison
        teile.append(spiele)

    # pd.concat haengt die Tabellen untereinander. ignore_index=True vergibt
    # neue, durchgehende Zeilennummern - wichtig, weil jede Datei wieder bei
    # 0 anfaengt und die Nummern sonst doppelt vorkaemen.
    alle_spiele = pd.concat(teile, ignore_index=True)

    # Nach Datum sortieren (stabil, siehe evaluate.py) und neu durchnummerieren.
    alle_spiele = alle_spiele.sort_values("datum", kind="stable").reset_index(drop=True)

    return alle_spiele, formate


def teams_der_saison(alle_spiele, saison):
    """Alle Teams einer Saison als Menge (set)."""
    return set(alle_spiele.loc[alle_spiele["saison"] == saison, "HomeTeam"])


def auf_und_absteiger(alle_spiele, saison):
    """
    Vergleicht eine Saison mit ihrer Vorsaison.
      Aufsteiger = Teams, die jetzt dabei sind, in der Vorsaison aber nicht.
      Absteiger  = Teams aus der Vorsaison, die jetzt nicht mehr dabei sind.

    Ist das ein Datenleck? Nein: Welche Teams in einer Saison spielen, steht
    vor dem ersten Spieltag fest. Es werden keine Ergebnisse benutzt.
    """
    jetzt = teams_der_saison(alle_spiele, saison)
    vorher = teams_der_saison(alle_spiele, vorsaison(saison))
    aufsteiger = sorted(jetzt - vorher)     # Mengen-Differenz: in jetzt, nicht in vorher
    absteiger = sorted(vorher - jetzt)
    return aufsteiger, absteiger


# ---------------------------------------------------------------------------
# 2. Datencheck
# ---------------------------------------------------------------------------

def hat_quoten(spiele):
    """True/False pro Spiel: Sind alle drei Pinnacle-Quoten vorhanden?"""
    return spiele[QUOTEN_SPALTEN].notna().all(axis=1).to_numpy()


def pruefe_daten(alle_spiele, formate):
    """
    Gibt einen Datencheck in der Konsole aus, BEVOR gerechnet wird.

    Warum? Fehler in den Daten fallen sonst erst als seltsame Ergebnisse auf -
    oder gar nicht. Typische Fallen bei mehreren Saisons: unvollstaendige
    Dateien, unterschiedliche Datumsformate, uneinheitlich geschriebene
    Teamnamen ("Man United" in einer Saison, "Man Utd" in einer anderen waeren
    fuer das Modell zwei verschiedene Teams) und fehlende Quoten.
    """
    print("Datencheck")
    print("=" * 72)

    # --- Spiele, Datumsformat, Zeitraum, fehlende Quoten pro Saison ---
    print(f"{'Saison':<9}{'Spiele':>7}  {'Datumsformat':<14}{'Zeitraum':<27}{'ohne Pinnacle':>13}")
    for saison in SAISONS:
        spiele = alle_spiele[alle_spiele["saison"] == saison]
        # Eine Saison hat 20 Teams, jedes spielt 2x gegen jedes andere: 20*19 = 380.
        hinweis = "" if len(spiele) == 380 else "  <- erwartet 380!"
        zeitraum = f"{spiele['datum'].min():%d.%m.%Y} - {spiele['datum'].max():%d.%m.%Y}"
        ohne_quoten = int((~hat_quoten(spiele)).sum())
        print(f"{saison_name(saison):<9}{len(spiele):>7}  {formate[saison]:<14}"
              f"{zeitraum:<27}{ohne_quoten:>13}{hinweis}")
    print()

    # --- Teamnamen ---
    # Zwei Pruefungen, die ein Computer gut kann:
    #   1. Leerzeichen am Anfang/Ende ("Arsenal " waere ein anderes Team)
    #   2. Namen, die sich nur in Gross-/Kleinschreibung oder Satzzeichen
    #      unterscheiden ("Nott'm Forest" gegen "Nottm Forest")
    # Abkuerzungen ("Man Utd" gegen "Man United") findet das nicht - deshalb
    # wird die komplette Liste zusaetzlich zum Ansehen ausgegeben.
    alle_namen = sorted(set(alle_spiele["HomeTeam"]) | set(alle_spiele["AwayTeam"]))
    mit_leerzeichen = [name for name in alle_namen if name != name.strip()]

    nach_vereinfachtem_namen = {}
    for name in alle_namen:
        vereinfacht = "".join(zeichen for zeichen in name.lower() if zeichen.isalnum())
        nach_vereinfachtem_namen.setdefault(vereinfacht, []).append(name)
    doppelt = [namen for namen in nach_vereinfachtem_namen.values() if len(namen) > 1]

    print(f"Teamnamen: {len(alle_namen)} verschiedene ueber alle Saisons")
    if not mit_leerzeichen and not doppelt:
        print("  Keine Leerzeichen-Fehler, keine Doppelschreibungen gefunden.")
    for name in mit_leerzeichen:
        print(f"  AUFFAELLIG: Leerzeichen in '{name}'")
    for namen in doppelt:
        print(f"  AUFFAELLIG: vermutlich dasselbe Team: {namen}")

    # Pro Team: in wie vielen der geladenen Saisons dabei?
    eintraege = []
    for name in alle_namen:
        anzahl = sum(name in teams_der_saison(alle_spiele, saison) for saison in SAISONS)
        eintraege.append(f"{name} ({anzahl})")
    print("  Alle Namen (Anzahl Saisons):")
    for start in range(0, len(eintraege), 5):
        print("    " + ", ".join(eintraege[start:start + 5]))
    print()

    # --- Auf- und Absteiger ---
    print("Auf- und Absteiger (gegenueber der Vorsaison):")
    for saison in SAISONS[1:]:
        aufsteiger, absteiger = auf_und_absteiger(alle_spiele, saison)
        print(f"  {saison_name(saison)}: Aufsteiger {', '.join(aufsteiger)}")
        print(f"  {'':7}  Absteiger der Vorsaison {', '.join(absteiger)}")
    print()

    # --- Fehlende Quoten (nur in den Testsaisons relevant) ---
    testspiele = alle_spiele[alle_spiele["saison"].isin(TESTSAISONS)]
    ohne = testspiele[~hat_quoten(testspiele)]
    print(f"Pinnacle-Quoten in den Testsaisons: {len(ohne)} Spiele ohne Quoten.")
    if len(ohne) > 0:
        print(f"  Zeitraum: {ohne['datum'].min():%d.%m.%Y} bis {ohne['datum'].max():%d.%m.%Y}")
        print("  Diese Spiele werden fuer ALLE Modelle aus der Bewertung genommen, damit")
        print("  alle Zeilen der Tabelle auf genau denselben Spielen beruhen.")
    print()


# ---------------------------------------------------------------------------
# 3. Walk-Forward
# ---------------------------------------------------------------------------

def trainingsspiele(alle_spiele, datum, saison, fenster):
    """
    Gibt die Trainingsspiele fuer ein Vorhersagedatum zurueck.

    Immer gilt: nur Spiele STRIKT vor dem Datum. Spiele vom selben Tag sind
    ausgeschlossen, weil man deren Ergebnis vor dem Anpfiff nicht kennt.

    fenster "A": zusaetzlich nur laufende und vorige Saison.
        Idee: Kader aendern sich jeden Sommer, alte Spiele sagen vielleicht
        wenig ueber die heutige Mannschaft.
    fenster "B": alle geladenen Spiele ab 2021/22.
        Idee: mehr Spiele = weniger Zufall in den Staerken.
    Welche Idee in der Praxis gewinnt, soll der Backtest zeigen.
    """
    bisher = alle_spiele[alle_spiele["datum"] < datum]

    if fenster == "A":
        erlaubte_saisons = [vorsaison(saison), saison]
        bisher = bisher[bisher["saison"].isin(erlaubte_saisons)]

    return bisher


def anzahl_spiele_im_training(training, team):
    """Wie viele Spiele (heim und auswaerts) hat ein Team im Trainingsfenster?"""
    return int(((training["HomeTeam"] == team) | (training["AwayTeam"] == team)).sum())


def ergaenze_aufsteiger(modell_heute, training, aufsteiger, absteiger, spiele_heute):
    """
    Passt die Staerken der Aufsteiger im Staerke-Dictionary an. Gibt zurueck,
    wie oft ein Team ganz ohne eigene Staerke den reinen Ersatzwert bekam
    (pro Team und Spiel).

    Das Problem: Ein Aufsteiger hat am Saisonanfang im Training keine oder nur
    sehr wenige Spiele - er hat ja in der zweiten Liga gespielt.
      - Ganz ohne Spiel fehlt er im Dictionary, und model.vorhersage() wuerde
        mit einem Fehler abbrechen.
      - Nach EINEM Spiel ist seine Staerke reiner Zufall. Ein Spiel ohne
        Gegentor ergibt Abwehrstaerke 0. Dann erwartet das Modell fuer den
        Gegner 0 Tore und gibt dessen Sieg exakt 0 % - der erste Lauf dieses
        Backtests hatte genau solche Prognosen.

    Der Ersatzwert: der Durchschnitt der drei Absteiger der Vorsaison, aus
    demselben Trainingsmodell. Begruendung: Aufsteiger ersetzen die Absteiger
    und sind meist ungefaehr so stark wie sie - deutlich schwaecher als der
    Liga-Schnitt. Den Liga-Schnitt (Staerke 1.0) zu nehmen, wuerde Aufsteiger
    systematisch ueberschaetzen.

    Shrinkage ("Schrumpfen" in Richtung des Ersatzwerts): Statt hart von
    Ersatzwert auf eigene Staerke umzuschalten, wird gleitend gemischt:

        staerke = (n * eigene_staerke + K * ersatzwert) / (n + K)

    n = Spiele des Teams im Trainingsfenster, K = SHRINKAGE_K = 10.
    Das ist ein gewichteter Durchschnitt, der Ersatzwert zaehlt wie K Spiele:
        n = 0  -> reiner Ersatzwert
        n = 1  -> 1/11 eigene Staerke, 10/11 Ersatzwert
        n = 10 -> je zur Haelfte
        n = 38 -> etwa 80 % eigene Staerke
    Je mehr eigene Spiele, desto mehr vertraut man den eigenen Daten. Eine
    einzelne Ausnahme-Partie kann die Staerke so nicht mehr auf 0 druecken.

    Das gilt nur fuer Aufsteiger der laufenden Saison (in der Vorsaison nicht
    in der Liga), getrennt fuer Angriff und Abwehr. Alle anderen Teams haben
    mindestens die volle Vorsaison im Training, bei ihnen ist das Problem
    nicht vorhanden. In Fenster B kann ein Aufsteiger auch Spiele aus
    frueheren Erstliga-Jahren haben (z.B. Leeds 2021-23); die zaehlen bei n mit.

    Warum hier und nicht in model.py? model.py soll unveraendert bleiben (die
    App benutzt es). Das Dictionary, das berechne_staerken() zurueckgibt, wird
    hier nur nachtraeglich angepasst.
    """
    staerken = modell_heute["staerken"]

    # Die Absteiger haben in der Vorsaison eine volle Saison gespielt und sind
    # in beiden Fenstern im Training. Fehlen sie trotzdem, stimmt etwas nicht.
    for team in absteiger:
        assert team in staerken, f"Absteiger {team} fehlt im Trainingsmodell"

    ersatzwert = {
        "angriff": np.mean([staerken[team]["angriff"] for team in absteiger]),
        "abwehr": np.mean([staerken[team]["abwehr"] for team in absteiger]),
    }

    # Shrinkage fuer Aufsteiger, die schon eine eigene Staerke haben.
    for team in aufsteiger:
        if team in staerken:
            n = anzahl_spiele_im_training(training, team)
            for art in ["angriff", "abwehr"]:
                eigene_staerke = staerken[team][art]
                staerken[team][art] = ((n * eigene_staerke + SHRINKAGE_K * ersatzwert[art])
                                       / (n + SHRINKAGE_K))

    # Teams, die heute spielen, aber gar keine eigene Staerke haben, bekommen
    # den reinen Ersatzwert - das entspricht der Formel mit n = 0.
    # Sonderfall: Ein Team, das im Training bisher NUR auswaerts gespielt hat,
    # fehlt ebenfalls im Dictionary, weil berechne_staerken() die Teamliste aus
    # den Heimteams bildet. Ohne eigene Staerke gibt es nichts zu mischen, also
    # auch hier der reine Ersatzwert. Das kommt nur ganz am Saisonanfang vor,
    # wo n so klein ist, dass die Formel ohnehin fast nur den Ersatzwert ergaebe.
    anzahl_ersetzt = 0
    teams_heute = list(spiele_heute["HomeTeam"]) + list(spiele_heute["AwayTeam"])
    for team in teams_heute:
        if team not in staerken:
            staerken[team] = dict(ersatzwert)   # dict(...) = eigene Kopie pro Team
            anzahl_ersetzt += 1

    return anzahl_ersetzt


def walk_forward_saison(alle_spiele, saison):
    """
    Sagt alle Spiele einer Testsaison Datum fuer Datum vorher.

    Gibt zurueck:
      - die Testspiele (in derselben Reihenfolge wie die Vorhersagen)
      - ein Dictionary Modellname -> Array mit Wahrscheinlichkeiten
      - ein Dictionary Fenster -> wie oft der Aufsteiger-Ersatzwert benutzt wurde
    """
    testspiele = alle_spiele[alle_spiele["saison"] == saison]
    aufsteiger, absteiger = auf_und_absteiger(alle_spiele, saison)

    zeilen = {BASISLINIE: [], POISSON_A: [], POISSON_B: []}
    ersatz_benutzt = {"A": 0, "B": 0}
    vorhergesagte_spiele = []

    # Wie in evaluate.py: groupby("datum") liefert die Daten aufsteigend
    # sortiert, pro Durchlauf ein Datum mit allen Spielen dieses Tages.
    # So wird pro Datum und Fenster genau einmal neu berechnet.
    for datum, spiele_heute in testspiele.groupby("datum"):
        for fenster, modellname in [("A", POISSON_A), ("B", POISSON_B)]:
            training = trainingsspiele(alle_spiele, datum, saison, fenster)

            # Sicherheitscheck gegen ein Datenleck, fuer jedes Datum und Fenster.
            assert training["datum"].max() < datum, f"Datenleck am {datum:%d.%m.%Y}!"

            modell_heute = model.berechne_staerken(ins_modellformat(training))
            ersatz_benutzt[fenster] += ergaenze_aufsteiger(
                modell_heute, training, aufsteiger, absteiger, spiele_heute)
            zeilen[modellname].extend(poisson_fuer_spiele(modell_heute, spiele_heute))

            # Basislinie: Ergebnis-Haeufigkeiten ALLER Spiele vor dem Datum,
            # also aus dem groessten Fenster B. Bei einer reinen Haeufigkeit
            # gibt es kein Problem mit veralteten Kadern, mehr Spiele machen
            # den Anteil nur stabiler.
            if fenster == "B":
                zeilen[BASISLINIE].extend(basislinie_vorhersagen(training, len(spiele_heute)))

        vorhergesagte_spiele.extend(spiele_heute.index)

    # Reihenfolge-Check wie in evaluate.py: Vorhersage Nummer i muss zu Spiel
    # Nummer i in testspiele gehoeren, sonst vergleichen die Kennzahlen
    # unbemerkt falsche Paare.
    assert vorhergesagte_spiele == list(testspiele.index), "Reihenfolge der Spiele stimmt nicht!"

    vorhersagen = {
        BASISLINIE: np.array(zeilen[BASISLINIE]),
        POISSON_A: np.array(zeilen[POISSON_A]),
        POISSON_B: np.array(zeilen[POISSON_B]),
        # Fuer Spiele ohne Quoten entstehen hier Zeilen mit "nan" (keine Zahl).
        # Die werden in nur_spiele_mit_quoten() herausgefiltert.
        PINNACLE: buchmacher_vorhersagen(testspiele),
    }
    return testspiele, vorhersagen, ersatz_benutzt


# ---------------------------------------------------------------------------
# 4. Bewertung
# ---------------------------------------------------------------------------

def nur_spiele_mit_quoten(testspiele, vorhersagen):
    """
    Behaelt nur die Spiele, fuer die Pinnacle-Quoten vorliegen - und zwar fuer
    ALLE Modelle gleich.

    Warum fuer alle? Wuerde man nur Pinnacle auf weniger Spielen bewerten,
    verglichen die Tabellenzeilen unterschiedliche Spiele miteinander, und die
    Unterschiede waeren nicht mehr nur den Modellen zuzuschreiben.
    """
    maske = hat_quoten(testspiele)
    wahr = tatsaechliche_ausgaenge(testspiele)[maske]

    gefiltert = {}
    for name in MODELLE:
        p = vorhersagen[name][maske]
        # Wie in evaluate.py: Jede Zeile muss eine vollstaendige Verteilung sein.
        assert np.allclose(p.sum(axis=1), 1.0), f"{name}: Zeilen summieren nicht zu 1"
        gefiltert[name] = p

    anzahl_ausgelassen = int((~maske).sum())
    return wahr, gefiltert, anzahl_ausgelassen


def log_loss_pro_spiel(p, wahr):
    """
    Log-Loss fuer jedes Spiel einzeln, statt als Durchschnitt.

    evaluate.log_loss() liefert nur den Durchschnitt. Fuer die Unsicherheit
    brauchen wir aber den Wert jedes einzelnen Spiels. Die Rechnung ist
    dieselbe (inklusive Untergrenze gegen ln(0)); der Durchschnitt dieser Werte
    wird unten mit evaluate.log_loss() abgeglichen.
    """
    p_wahrer_ausgang = np.clip(np.sum(p * wahr, axis=1), 1e-15, 1.0)
    return -np.log(p_wahrer_ausgang)


def differenz_mit_unsicherheit(p_1, p_2, wahr):
    """
    Wie sicher ist ein Unterschied im Log-Loss zwischen zwei Modellen?

    Vorgehen: Fuer jedes Spiel die Differenz Log-Loss(Modell 1) minus
    Log-Loss(Modell 2) bilden. Davon den Mittelwert und den Standardfehler:

        Standardfehler = Standardabweichung der Differenzen / Wurzel(Anzahl Spiele)
        Bereich        = Mittelwert +/- 1.96 * Standardfehler

    Warum gepaart (Spiel fuer Spiel)? Beide Modelle sagen dieselben Spiele
    vorher. Manche Spiele sind fuer jedes Modell schwer (Ueberraschungen).
    Vergleicht man pro Spiel, faellt diese gemeinsame Schwierigkeit heraus und
    uebrig bleibt der echte Unterschied zwischen den Modellen.

    Lesart: negativ = Modell 1 besser, positiv = Modell 2 besser. Enthaelt der
    Bereich die 0, ist der Unterschied nicht von Zufall zu unterscheiden.

    ACHTUNG, grobe Faustregel und kein formaler Test: Die 1.96 stammen aus der
    Normalverteilung (ca. 95 %), und es wird angenommen, dass die Spiele
    voneinander unabhaengig sind. Das stimmt nur ungefaehr - dieselben Teams
    spielen immer wieder. Der echte Unsicherheitsbereich ist eher etwas breiter.
    """
    werte_1 = log_loss_pro_spiel(p_1, wahr)
    werte_2 = log_loss_pro_spiel(p_2, wahr)

    # Abgleich mit evaluate.log_loss(): beide muessen dasselbe ergeben.
    assert np.isclose(np.mean(werte_1), log_loss(p_1, wahr))
    assert np.isclose(np.mean(werte_2), log_loss(p_2, wahr))

    differenzen = werte_1 - werte_2
    standardfehler = np.std(differenzen, ddof=1) / np.sqrt(len(differenzen))
    return np.mean(differenzen), 1.96 * standardfehler


# ---------------------------------------------------------------------------
# 5. Ausgabe
# ---------------------------------------------------------------------------

def anzahl_unter_einem_prozent(p, wahr):
    """
    Bei wie vielen Spielen hat das Modell dem Ausgang, der wirklich eintrat,
    weniger als 1 % gegeben?

    Warum fest in der Ausgabe? Solche Prognosen sind fast immer ein Zeichen
    fuer einen Fehler (z.B. Staerke 0 nach einem einzigen Spiel), kein
    "mutiger Tipp". Beim Log-Loss koennen wenige davon das Ergebnis einer
    ganzen Saison bestimmen. Mit dieser Zahl in der Tabelle faellt das sofort
    auf, statt sich hinter einem Durchschnitt zu verstecken.
    """
    p_wahrer_ausgang = np.sum(p * wahr, axis=1)
    return int(np.sum(p_wahrer_ausgang < 0.01))


def tabellenzeilen(bezeichnung, wahr, vorhersagen):
    """Kennzahlen aller Modelle fuer eine Saison (oder alle zusammen)."""
    zeilen = []
    for name in MODELLE:
        p = vorhersagen[name]
        zeilen.append((bezeichnung, name, len(wahr),
                       accuracy(p, wahr), log_loss(p, wahr), brier_score(p, wahr),
                       anzahl_unter_einem_prozent(p, wahr)))
    return zeilen


def unsicherheits_text(titel, mittel, bereich):
    """Ein Satz zur Log-Loss-Differenz, fuer Konsole und Markdown gleich."""
    enthaelt_null = (mittel - bereich) <= 0 <= (mittel + bereich)
    lesart = "nicht von Zufall zu unterscheiden" if enthaelt_null else "Bereich enthaelt die 0 nicht"
    return f"{titel}: {mittel:+.4f} +/- {bereich:.4f} ({lesart})"


def drucke_ergebnisse(zeilen, ersatz_texte, unsicherheit_texte):
    print("Walk-Forward-Backtest")
    print("=" * 72)
    print(f"{'Saison':<9}{'Modell':<34}{'Spiele':>7}{'Accuracy':>10}{'Log-Loss':>10}"
          f"{'Brier':>8}{'<1%':>6}")
    vorherige_saison = None
    for saison, name, anzahl, acc, ll, bs, unter_1 in zeilen:
        if vorherige_saison is not None and saison != vorherige_saison:
            print()
        vorherige_saison = saison
        print(f"{saison:<9}{name:<34}{anzahl:>7}{acc * 100:>8.1f} %{ll:>10.3f}{bs:>8.3f}"
              f"{unter_1:>6}")
    print()
    print("Accuracy: hoeher ist besser. Log-Loss und Brier-Score: niedriger ist besser.")
    print("<1%: Anzahl Spiele, bei denen das Modell dem echten Ausgang unter 1 % gab.")
    print()
    print("Reiner Ersatzwert benutzt, Team ganz ohne eigene Staerke (Anzahl Team-Einsaetze):")
    for text in ersatz_texte:
        print(f"  {text}")
    print()
    print("Log-Loss-Differenz pro Spiel, alle Testsaisons, Mittelwert +/- 1,96 Standardfehler")
    print("(negativ = erstes Modell besser; grobe Faustregel, kein formaler Test):")
    for text in unsicherheit_texte:
        print(f"  {text}")


def schreibe_markdown(pfad, zeilen, ersatz_texte, unsicherheit_texte, ausgelassen_texte):
    os.makedirs(os.path.dirname(pfad), exist_ok=True)

    md = ["# Backtest ueber mehrere Saisons (Walk-Forward)", ""]
    md.append("Daten: Premier League von football-data.co.uk, Saisons "
              + ", ".join(saison_name(s) for s in SAISONS)
              + ". 2019/20 und 2020/21 bewusst ausgelassen (Corona, kaum Heimvorteil).")
    md.append("")
    md.append("Vorhergesagt werden die Saisons "
              + ", ".join(saison_name(s) for s in TESTSAISONS)
              + ". Vor jedem Spieltag werden die Staerken aus allen Spielen strikt vor "
              "diesem Datum neu berechnet:")
    md.append("")
    md.append("- **Fenster A (Aktuell + Vorsaison):** nur laufende und vorige Saison")
    md.append("- **Fenster B (Alle bisherigen):** alle Spiele ab 2021/22")
    md.append("- **Basislinie:** Ergebnis-Haeufigkeiten aller Spiele vor dem Datum")
    md.append("- **Pinnacle:** 1/Quote, normalisiert (ohne Marge)")
    md.append("")
    md.append("Aufsteiger: Ersatzwert = Durchschnitt der drei Absteiger der Vorsaison. "
              f"Gemischt mit der eigenen Staerke als (n * eigene + {SHRINKAGE_K} * Ersatzwert) "
              f"/ (n + {SHRINKAGE_K}), n = eigene Spiele im Trainingsfenster. Ohne eigene "
              "Spiele gilt der reine Ersatzwert.")
    md.append("")
    md.append("Spiele ohne Pinnacle-Quoten sind fuer alle Modelle aus der Bewertung genommen:")
    md.append("")
    for text in ausgelassen_texte:
        md.append(f"- {text}")
    md.append("")
    md.append("| Saison | Modell | Spiele | Accuracy | Log-Loss | Brier-Score | P(echt) < 1 % |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for saison, name, anzahl, acc, ll, bs, unter_1 in zeilen:
        md.append(f"| {saison} | {name} | {anzahl} | {acc * 100:.1f} % | {ll:.3f} | "
                  f"{bs:.3f} | {unter_1} |")
    md.append("")
    md.append("Accuracy: hoeher ist besser. Log-Loss und Brier-Score: niedriger ist besser. "
              "P(echt) < 1 %: Anzahl Spiele, bei denen das Modell dem echten Ausgang "
              "weniger als 1 % gab.")
    md.append("")
    md.append("**Reiner Ersatzwert benutzt**, Team ganz ohne eigene Staerke (Anzahl Team-Einsaetze):")
    md.append("")
    for text in ersatz_texte:
        md.append(f"- {text}")
    md.append("")
    md.append("**Log-Loss-Differenz pro Spiel**, alle Testsaisons, Mittelwert +/- 1,96 "
              "Standardfehler. Negativ = erstes Modell besser. Grobe Faustregel, kein "
              "formaler Test (Spiele sind nicht voellig unabhaengig voneinander).")
    md.append("")
    for text in unsicherheit_texte:
        md.append(f"- {text}")
    md.append("")
    md.append("Erzeugt mit `python3 lade_daten.py` und `python3 backtest_saisons.py`.")

    with open(pfad, "w", encoding="utf-8") as datei:
        datei.write("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# Programmablauf
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    alle_spiele, formate = lade_alle_saisons()
    pruefe_daten(alle_spiele, formate)

    zeilen = []
    ersatz_texte = []
    ausgelassen_texte = []

    # Fuer die Gesamtwertung werden die bewerteten Spiele aller Testsaisons
    # gesammelt und am Ende zusammengehaengt.
    alle_wahr = []
    alle_vorhersagen = {name: [] for name in MODELLE}

    for saison in TESTSAISONS:
        testspiele, vorhersagen, ersatz_benutzt = walk_forward_saison(alle_spiele, saison)
        wahr, gefiltert, ausgelassen = nur_spiele_mit_quoten(testspiele, vorhersagen)

        zeilen.extend(tabellenzeilen(saison_name(saison), wahr, gefiltert))
        ersatz_texte.append(f"{saison_name(saison)}: Fenster A {ersatz_benutzt['A']}x, "
                            f"Fenster B {ersatz_benutzt['B']}x")
        ausgelassen_texte.append(f"{saison_name(saison)}: {ausgelassen} von "
                                 f"{len(testspiele)} Spielen ausgelassen")

        alle_wahr.append(wahr)
        for name in MODELLE:
            alle_vorhersagen[name].append(gefiltert[name])

    # np.concatenate haengt die Arrays der einzelnen Saisons untereinander.
    gesamt_wahr = np.concatenate(alle_wahr)
    gesamt = {name: np.concatenate(alle_vorhersagen[name]) for name in MODELLE}
    zeilen.extend(tabellenzeilen("Gesamt", gesamt_wahr, gesamt))

    # Unsicherheit 1: Fenster A gegen Fenster B.
    mittel_ab, bereich_ab = differenz_mit_unsicherheit(gesamt[POISSON_A], gesamt[POISSON_B], gesamt_wahr)

    # Unsicherheit 2: das bessere Fenster (niedrigerer Gesamt-Log-Loss) gegen Pinnacle.
    if log_loss(gesamt[POISSON_A], gesamt_wahr) <= log_loss(gesamt[POISSON_B], gesamt_wahr):
        bestes = POISSON_A
    else:
        bestes = POISSON_B
    mittel_bp, bereich_bp = differenz_mit_unsicherheit(gesamt[bestes], gesamt[PINNACLE], gesamt_wahr)

    unsicherheit_texte = [
        unsicherheits_text("Poisson A minus Poisson B", mittel_ab, bereich_ab),
        unsicherheits_text(f"{bestes} minus Pinnacle", mittel_bp, bereich_bp),
    ]

    drucke_ergebnisse(zeilen, ersatz_texte, unsicherheit_texte)
    schreibe_markdown(ERGEBNIS_PFAD, zeilen, ersatz_texte, unsicherheit_texte, ausgelassen_texte)
    print()
    print(f"Gespeichert in {ERGEBNIS_PFAD}")
