# model.py
# Hier liegt die gesamte Vorhersage-Logik (die "Mathematik").
# Diese Datei weiss NICHTS vom Web/Flask. Man kann sie alleine testen.

import math

import pandas as pd


def lade_spiele(pfad):
    """
    Liest die Saison-CSV ein und gibt eine Tabelle (DataFrame) zurueck,
    die nur die vier Spalten enthaelt, die wir wirklich brauchen.

    pfad: Dateipfad zur CSV, z.B. "data/E0.csv"
    """

    # Die komplette CSV einlesen. Sie hat ueber 100 Spalten -
    # die meisten (Wettquoten, Karten, ...) interessieren uns nicht.
    alle_daten = pd.read_csv(pfad)

    # Wir behalten nur die vier relevanten Spalten:
    #   HomeTeam = Heimteam
    #   AwayTeam = Auswaertsteam
    #   FTHG     = Full Time Home Goals (Endstand Heimtore)
    #   FTAG     = Full Time Away Goals (Endstand Auswaertstore)
    spiele = alle_daten[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]

    # Wir benennen die Spalten in verstaendliche deutsche Namen um.
    # Ab jetzt heisst die Spalte z.B. "heimtore" statt "FTHG".
    spiele = spiele.rename(columns={
        "HomeTeam": "heim",
        "AwayTeam": "auswaerts",
        "FTHG": "heimtore",
        "FTAG": "auswaertstore",
    })

    return spiele


def berechne_staerken(spiele):
    """
    Berechnet aus den Spieldaten:
      - den Liga-Durchschnitt der Heim- und der Auswaertstore
      - pro Team eine Angriffs- und eine Abwehrstaerke

    Gibt alles gebuendelt in einem Dictionary zurueck, damit die
    Vorhersage-Funktion spaeter alles aus einer Hand bekommt.
    """

    # --- 1. Liga-Durchschnitte (Heim und Auswaerts getrennt) ---
    # mean() rechnet den Durchschnitt einer Spalte aus.
    # liga_heim_schnitt = "Was schiesst ein durchschnittliches Heimteam?"
    liga_heim_schnitt = spiele["heimtore"].mean()
    # liga_auswaerts_schnitt = "Was schiesst ein durchschnittliches Auswaertsteam?"
    liga_auswaerts_schnitt = spiele["auswaertstore"].mean()

    # Fuer die EINE Angriffs-/Abwehrstaerke pro Team brauchen wir einen
    # gemeinsamen Vergleichswert: die durchschnittlichen Tore PRO TEAM PRO SPIEL.
    # Insgesamt fallen (Heimtore + Auswaertstore) Tore, verteilt auf
    # 2 Teams pro Spiel -> deshalb durch (2 * Anzahl Spiele) teilen.
    anzahl_spiele = len(spiele)
    alle_tore = spiele["heimtore"].sum() + spiele["auswaertstore"].sum()
    liga_schnitt_pro_team = alle_tore / (2 * anzahl_spiele)

    # --- 2. Angriffs- und Abwehrstaerke pro Team ---
    # Wir sammeln die Ergebnisse in einem Dictionary:
    #   staerken["Arsenal"] = {"angriff": 1.4, "abwehr": 0.7}
    staerken = {}

    # Jede Mannschaft taucht in einer vollen Saison als Heimteam auf,
    # daher liefert die Spalte "heim" alle Teamnamen. unique() = ohne Doppelte.
    teams = sorted(spiele["heim"].unique())

    for team in teams:
        # Alle Heimspiele und alle Auswaertsspiele dieses Teams herausfiltern.
        heimspiele = spiele[spiele["heim"] == team]
        auswaertsspiele = spiele[spiele["auswaerts"] == team]

        # Geschossene Tore = zuhause geschossen + auswaerts geschossen.
        geschossen = heimspiele["heimtore"].sum() + auswaertsspiele["auswaertstore"].sum()
        # Kassierte Tore = zuhause kassiert (das sind die Auswaertstore des Gegners)
        #                  + auswaerts kassiert (die Heimtore des Gegners).
        kassiert = heimspiele["auswaertstore"].sum() + auswaertsspiele["heimtore"].sum()

        # Wie viele Spiele hat das Team insgesamt bestritten? (sollte 38 sein)
        gespielt = len(heimspiele) + len(auswaertsspiele)

        # Durchschnitt pro Spiel fuer dieses Team.
        schnitt_geschossen = geschossen / gespielt
        schnitt_kassiert = kassiert / gespielt

        # Staerke = Team-Schnitt im Verhaeltnis zum Liga-Schnitt.
        # angriff > 1 -> schiesst mehr als der Schnitt (gut)
        # abwehr  < 1 -> kassiert weniger als der Schnitt (gut)
        angriff = schnitt_geschossen / liga_schnitt_pro_team
        abwehr = schnitt_kassiert / liga_schnitt_pro_team

        staerken[team] = {"angriff": angriff, "abwehr": abwehr}

    # Alles gebuendelt zurueckgeben.
    return {
        "liga_heim_schnitt": liga_heim_schnitt,
        "liga_auswaerts_schnitt": liga_auswaerts_schnitt,
        "staerken": staerken,
    }


def poisson_wahrscheinlichkeit(k, lam):
    """
    Die Poisson-Formel: Wahrscheinlichkeit fuer GENAU k Tore,
    wenn im Schnitt lam (Lambda) Tore erwartet werden.

        P(k) = (lam^k * e^(-lam)) / k!

    k:   die konkrete Toranzahl (0, 1, 2, ...)
    lam: der erwartete Durchschnitt (z.B. 1.7)
    """
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def erwartete_tore(modell, heim, auswaerts):
    """
    Berechnet die erwarteten Tore beider Teams fuer ein Spiel.
    Formel (aus Schritt 3):
        erwartete Heimtore   = Angriff(Heim)   * Abwehr(Auswaerts) * Liga-Heim-Schnitt
        erwartete Auswaertst.= Angriff(Auswaerts) * Abwehr(Heim)   * Liga-Auswaerts-Schnitt
    """
    staerken = modell["staerken"]

    # Die vier benoetigten Staerkewerte herausholen.
    angriff_heim = staerken[heim]["angriff"]
    abwehr_heim = staerken[heim]["abwehr"]
    angriff_auswaerts = staerken[auswaerts]["angriff"]
    abwehr_auswaerts = staerken[auswaerts]["abwehr"]

    erwartete_heimtore = angriff_heim * abwehr_auswaerts * modell["liga_heim_schnitt"]
    erwartete_auswaertstore = angriff_auswaerts * abwehr_heim * modell["liga_auswaerts_schnitt"]

    return erwartete_heimtore, erwartete_auswaertstore


def vorhersage(modell, heim, auswaerts, max_tore=10):
    """
    Die Hauptfunktion: berechnet fuer ein Spiel
      - die Wahrscheinlichkeit fuer Heimsieg / Remis / Auswaertssieg
      - die wahrscheinlichsten Ergebnisse (Top 4)

    max_tore: bis zu wie vielen Toren pro Team wir rechnen.
              Je hoeher, desto naeher summieren sich die Wahrscheinlichkeiten
              an 100 %. 10 ist ein guter Kompromiss: hoehere Ergebnisse sind
              so unwahrscheinlich, dass sie praktisch nichts mehr aendern.
    """

    # 1. Erwartete Tore (Lambda) fuer beide Teams.
    lam_heim, lam_auswaerts = erwartete_tore(modell, heim, auswaerts)

    # 2. Wir laufen ueber das Raster aller Ergebnisse und summieren auf.
    p_heimsieg = 0.0
    p_remis = 0.0
    p_auswaertssieg = 0.0
    ergebnisse = []  # Liste von (heimtore, auswaertstore, wahrscheinlichkeit)

    for i in range(max_tore + 1):          # i = Tore des Heimteams
        for j in range(max_tore + 1):      # j = Tore des Auswaertsteams
            # Wahrscheinlichkeit fuer genau dieses Ergebnis i:j.
            # Beide Teams als unabhaengig angenommen -> multiplizieren.
            p = poisson_wahrscheinlichkeit(i, lam_heim) * poisson_wahrscheinlichkeit(j, lam_auswaerts)

            ergebnisse.append((i, j, p))

            # Je nach Ausgang in den passenden Topf addieren.
            if i > j:
                p_heimsieg += p
            elif i == j:
                p_remis += p
            else:
                p_auswaertssieg += p

    # 3. Die wahrscheinlichsten Ergebnisse herausfinden:
    #    nach der Wahrscheinlichkeit (3. Element, Index 2) absteigend sortieren.
    ergebnisse.sort(key=lambda eintrag: eintrag[2], reverse=True)
    top_ergebnisse = ergebnisse[:4]

    return {
        "erwartete_heimtore": lam_heim,
        "erwartete_auswaertstore": lam_auswaerts,
        "p_heimsieg": p_heimsieg,
        "p_remis": p_remis,
        "p_auswaertssieg": p_auswaertssieg,
        "top_ergebnisse": top_ergebnisse,
    }


# Dieser Block laeuft NUR, wenn man die Datei direkt startet
# (also "python model.py"). Wenn Flask spaeter model.py importiert,
# wird dieser Block NICHT ausgefuehrt. Praktisch zum Testen.
if __name__ == "__main__":
    spiele = lade_spiele("data/E0.csv")
    modell = berechne_staerken(spiele)

    heim = "Man City"
    auswaerts = "Burnley"
    ergebnis = vorhersage(modell, heim, auswaerts)

    print(f"Vorhersage: {heim} (Heim) gegen {auswaerts} (Auswaerts)")
    print(f"  Erwartete Tore: {ergebnis['erwartete_heimtore']:.2f} : {ergebnis['erwartete_auswaertstore']:.2f}")
    print()
    print(f"  Heimsieg:       {ergebnis['p_heimsieg'] * 100:5.1f} %")
    print(f"  Unentschieden:  {ergebnis['p_remis'] * 100:5.1f} %")
    print(f"  Auswaertssieg:  {ergebnis['p_auswaertssieg'] * 100:5.1f} %")
    print()
    print("  Wahrscheinlichste Ergebnisse:")
    for heimtore, auswaertstore, p in ergebnis["top_ergebnisse"]:
        print(f"    {heimtore}:{auswaertstore}  ({p * 100:.1f} %)")
