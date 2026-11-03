#!/usr/bin/env python3
"""Fasst die Ergebnisse ueber alle Datensaetze zu einer Rangfolge zusammen.

Jeder Datensatz wird mit seiner eigenen Kaggle-Metrik bewertet, und zehn
verschiedene Metriken lassen sich nicht mitteln - ein AUC von 0,9 und ein RMSE
von 87534 haben keinen gemeinsamen Nenner. Vergleichbar sind dagegen die
*Raenge*: Pro Datensatz bekommt das beste Modell Rang 1, das zweitbeste Rang 2.
Ueber die Datensaetze hinweg wird dann der mittlere Rang gebildet.

Dafuer ist die Spalte 'richtung' in der Ergebnistabelle da: Bei AUC ist gross
gut, bei RMSE klein.

Danach die ueblichen Tests fuer genau diesen Fall (Demsar 2006, "Statistical
Comparisons of Classifiers over Multiple Data Sets", JMLR):

    Friedman   Unterscheiden sich die Modelle ueberhaupt? (ab 3 Modellen)
    Wilcoxon   dasselbe fuer genau 2 Modelle
    Nemenyi    welche Paare unterscheiden sich? Ueber die kritische
               Differenz CD: Zwei Modelle sind verschieden, wenn ihre
               mittleren Raenge weiter als CD auseinanderliegen.

    python scripts/aggregate_ranks.py
    python scripts/aggregate_ranks.py --variante voll
    python scripts/aggregate_ranks.py --modelle rf xgb tabpfn
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
ERGEBNISSE = ROOT / "results" / "model_results.csv"
AUSGABE_DIR = ROOT / "results"

# Kritische Werte der Nemenyi-Statistik fuer alpha = 0,05, nach Demsar (2006),
# Tabelle 5. Schluessel ist die Anzahl verglichener Modelle.
Q_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def lade_ergebnisse(pfad, variante, modelle):
    """Ergebnistabelle als slug x modell, nur vollstaendige Datensaetze."""
    roh = pd.read_csv(pfad, sep=";", encoding="utf-8-sig")

    gebraucht = {"slug", "modell", "variante", "status", "wert_mittel",
                 "richtung", "metrik_code"}
    fehlt = gebraucht - set(roh.columns)
    if fehlt:
        raise SystemExit("In {} fehlen Spalten: {}".format(pfad, ", ".join(sorted(fehlt))))

    ok = roh[(roh["status"] == "ok") & (roh["variante"] == variante)]
    if modelle:
        ok = ok[ok["modell"].isin(modelle)]
    if ok.empty:
        raise SystemExit("Keine Ergebnisse mit status=ok fuer Variante " + variante)

    # Aeltere Ergebnisdateien haben die Spalte noch nicht.
    umgebungen = (sorted(ok["umgebung"].dropna().unique())
                  if "umgebung" in ok.columns else [])
    werte = ok.pivot_table(index="slug", columns="modell", values="wert_mittel")
    richtung = ok.groupby("slug")["richtung"].first()
    metrik = ok.groupby("slug")["metrik_code"].first()

    # Friedman und Nemenyi brauchen fuer jeden Datensatz jedes Modell. Ein
    # Datensatz, bei dem ein Modell fehlt, wuerde die Raenge verschieben.
    vollstaendig = werte.dropna()
    fehlend = werte.index.difference(vollstaendig.index)
    return vollstaendig, richtung, metrik, list(fehlend), umgebungen


def raenge(werte, richtung):
    """Rang 1 ist das beste Modell, Bindungen bekommen den Mittelwert."""
    aus = pd.DataFrame(index=werte.index, columns=werte.columns, dtype=float)
    for slug in werte.index:
        # richtung -1 heisst 'kleiner ist besser', dann aufsteigend ranken.
        aufsteigend = richtung[slug] < 0
        aus.loc[slug] = werte.loc[slug].rank(ascending=aufsteigend, method="average")
    return aus


def kritische_differenz(k, n):
    """CD der Nemenyi-Statistik, oder None wenn k nicht in der Tabelle steht."""
    if k not in Q_005:
        return None
    return Q_005[k] * np.sqrt(k * (k + 1) / (6.0 * n))


def gruppen(mittel, cd):
    """Maximale Gruppen von Modellen, die sich nicht unterscheiden.

    Im Diagramm sind das die Balken: Modelle unter einem Balken liegen enger
    als CD zusammen, ihr Unterschied ist also nicht belegt.
    """
    sortiert = mittel.sort_values()
    namen = list(sortiert.index)
    aus = []
    for i in range(len(namen)):
        j = i
        while j + 1 < len(namen) and sortiert.iloc[j + 1] - sortiert.iloc[i] < cd:
            j += 1
        if j > i and not any(i >= a and j <= b for a, b in aus):
            aus.append((i, j))
    return [[namen[x] for x in range(a, b + 1)] for a, b in aus]


def diagramm(mittel, cd, n, pfad):
    """Kritische-Differenz-Diagramm, so wie es bei Demsar abgebildet ist."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib fehlt, kein Diagramm geschrieben")
        return None

    sortiert = mittel.sort_values()
    k = len(sortiert)

    # Von oben nach unten: Achse, CD-Balken, dann je Modell eine Stiellinie,
    # ganz unten die Gruppenbalken. Jede Ebene bekommt eigenen Platz, sonst
    # ueberschreiben sich Beschriftungen.
    y_achse = 0.0
    y_erster_stiel = 1.0
    abstand = 0.75
    gruppen_liste = gruppen(sortiert, cd) if cd is not None else []
    y_gruppen = y_erster_stiel + abstand * (k - 1) + 0.6
    # Der CD-Balken bekommt eine eigene Zeile ganz unten, sonst kreuzen ihn
    # die Stiellinien der Modelle.
    y_cd = y_gruppen + 0.45 * max(len(gruppen_liste), 1) + 0.55
    hoehe_gesamt = y_cd + 0.75

    fig, ax = plt.subplots(figsize=(8.5, 1.05 + 0.52 * hoehe_gesamt))
    links, rechts = 1, k
    rand = 0.12 * (rechts - links) + 0.3
    ax.set_xlim(rechts + rand, links - rand)  # Rang 1 nach rechts, wie ueblich
    ax.set_ylim(hoehe_gesamt, -0.6)  # y waechst nach unten
    ax.spines[["left", "right", "bottom"]].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(links, rechts + 1))
    ax.set_xlabel("mittlerer Rang (1 = beste Leistung)")

    for i, (name, wert) in enumerate(sortiert.items()):
        y = y_erster_stiel + abstand * i
        ax.plot([wert, wert], [y_achse, y], color="0.45", linewidth=0.9)
        ax.plot([wert], [y_achse], marker="o", markersize=4, color="0.2")
        # Links von der Mitte nach links beschriften, rechts nach rechts -
        # so laeuft keine Beschriftung aus dem Bild.
        nach_links = wert > (links + rechts) / 2
        ax.annotate(
            "{}  ({:.2f})".format(name, wert), xy=(wert, y),
            xytext=(-6 if nach_links else 6, 0), textcoords="offset points",
            ha="right" if nach_links else "left", va="center", fontsize=10,
        )

    for nummer, gruppe in enumerate(gruppen_liste):
        y = y_gruppen + 0.45 * nummer
        ax.plot([sortiert[gruppe[0]], sortiert[gruppe[-1]]], [y, y],
                color="0.1", linewidth=4.5, solid_capstyle="round")

    if cd is not None:
        # Am linken Rand verankert, damit der Balken als Massstab gelesen wird
        # und nicht als Aussage ueber bestimmte Modelle.
        von, bis = rechts, rechts - cd
        ax.plot([von, bis], [y_cd, y_cd], color="0.2", linewidth=2)
        for x in (von, bis):
            ax.plot([x, x], [y_cd - 0.13, y_cd + 0.13], color="0.2", linewidth=2)
        ax.text((von + bis) / 2, y_cd + 0.22, "CD = {:.2f}".format(cd),
                ha="center", va="top", fontsize=10)

    ax.set_title("{} Modelle, {} Datensaetze".format(k, n), fontsize=10, pad=26)
    fig.tight_layout()
    pfad.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pfad, dpi=150)
    plt.close(fig)
    return pfad


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ergebnisse", default=str(ERGEBNISSE))
    parser.add_argument("--variante", default="n10000")
    parser.add_argument("--modelle", nargs="+", help="nur diese Modelle vergleichen")
    parser.add_argument("--ausgabe", default=str(AUSGABE_DIR))
    args = parser.parse_args()

    werte, richtung, metrik, fehlend, umgebungen = lade_ergebnisse(
        Path(args.ergebnisse), args.variante, args.modelle
    )
    rang = raenge(werte, richtung)
    mittel = rang.mean().sort_values()
    k, n = len(mittel), len(rang)

    print("Variante {}: {} Modelle auf {} Datensaetzen".format(args.variante, k, n))
    if len(umgebungen) > 1:
        print("WARNUNG: Ergebnisse aus mehreren Umgebungen, Zeiten nicht vergleichbar:")
        for u in umgebungen:
            print("   ", u)
    if fehlend:
        print("ausgelassen, weil ein Modell fehlt ({}): {}".format(
            len(fehlend), ", ".join(s.replace("playground-series-", "") for s in fehlend)))
    print()

    print("mittlerer Rang (1 = beste Leistung)")
    for name, wert in mittel.items():
        siege = int((rang[name] == rang.min(axis=1)).sum())
        print("  {:<10} {:.3f}   bester oder gleichauf auf {:>2} von {} Datensaetzen".format(
            name, wert, siege, n))
    print()

    if k < 2:
        raise SystemExit("Fuer einen Vergleich braucht es mindestens 2 Modelle.")

    if k == 2:
        a, b = mittel.index
        try:
            stat, p = wilcoxon(werte[a], werte[b])
            print("Wilcoxon (2 Modelle): W = {:.1f}, p = {:.4g}".format(stat, p))
        except ValueError as exc:
            stat, p = float("nan"), float("nan")
            print("Wilcoxon nicht berechenbar: " + str(exc)[:80])
    else:
        stat, p = friedmanchisquare(*[rang[m] for m in mittel.index])
        print("Friedman: chi2 = {:.3f}, p = {:.4g}".format(stat, p))

    # Bei lauter Gleichstaenden liefern beide Tests nan. Ein Vergleich wie
    # 'nan < 0.05' ist immer falsch und wuerde hier stillschweigend einen
    # belegten Unterschied behaupten.
    if np.isnan(p):
        print("  Kein Testergebnis - die Modelle liegen auf allen Datensaetzen gleich.")
    elif p < 0.05:
        print("  Mindestens zwei Modelle unterscheiden sich (p < 0,05).")
    else:
        print("  Kein belegter Unterschied zwischen den Modellen (p >= 0,05).")
        print("  Ein Nemenyi-Nachtest ist dann nicht angebracht.")

    cd = kritische_differenz(k, n)
    if cd is None:
        print("\nKein Nemenyi-Wert fuer {} Modelle in der Tabelle.".format(k))
    else:
        print("\nNemenyi, alpha = 0,05: CD = {:.3f}".format(cd))
        for i, a in enumerate(mittel.index):
            for b in list(mittel.index)[i + 1:]:
                diff = abs(mittel[a] - mittel[b])
                zeichen = "unterscheidet sich" if diff > cd else "nicht belegt"
                print("  {:<10} vs {:<10} Differenz {:.3f}   {}".format(a, b, diff, zeichen))

    ausgabe = Path(args.ausgabe)
    ausgabe.mkdir(parents=True, exist_ok=True)

    tabelle = rang.copy()
    tabelle.insert(0, "metrik_code", metrik.reindex(tabelle.index))
    tabelle.insert(1, "richtung", richtung.reindex(tabelle.index))
    for name in werte.columns:
        tabelle["wert_" + name] = werte[name]
    datei = ausgabe / "ranks_{}.csv".format(args.variante)
    tabelle.to_csv(datei, sep=";", encoding="utf-8-sig")

    bild = diagramm(mittel, cd, n, ausgabe / "cd_{}.png".format(args.variante))
    print("\nRaenge  -> " + str(datei))
    if bild:
        print("Diagramm -> " + str(bild))


if __name__ == "__main__":
    main()
