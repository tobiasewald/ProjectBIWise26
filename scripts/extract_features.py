#!/usr/bin/env python3
"""Extrahiert Basis-Merkmale aus den heruntergeladenen Datensaetzen.

Liest je Wettbewerb data/<slug>/train.csv und schreibt eine Zeile nach
metadata/dataset_features.csv. Ueber die Spalte 'slug' laesst sich die Datei
mit der Uebersichtstabelle verknuepfen.

Die Zielspalte wird bestimmt als die Spalte, die in train.csv steht, in
test.csv aber fehlt - bei den Playground-Wettbewerben ist das zuverlaessig.

    python scripts/extract_features.py
    python scripts/extract_features.py --only s4e1
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_FILE = ROOT / "metadata" / "dataset_features.csv"

COLUMNS = [
    "slug",
    "n_zeilen",
    "n_spalten",
    "n_features",
    "zielspalte",
    "ziel_dtype",
    "ziel_n_unique",
    "aufgabe_vermutet",
    "n_numerisch",
    "n_kategorial",
    "n_fast_eindeutig",
    "n_fehlende_werte",
    "anteil_fehlend_prozent",
    "spalten_mit_fehlenden",
    "train_mb",
]


def find_targets(train, slug):
    """Spalten, die in train.csv stehen, in test.csv aber fehlen."""
    test_path = DATA_DIR / slug / "test.csv"
    if not test_path.is_file():
        return []
    test_columns = pd.read_csv(test_path, nrows=0).columns
    return [c for c in train.columns if c not in test_columns]


def guess_task(series):
    """Grobe Einordnung der Aufgabe anhand der Zielspalte."""
    n_unique = series.nunique(dropna=True)
    if n_unique == 2:
        return "binaer"
    if pd.api.types.is_numeric_dtype(series) and n_unique > 20:
        return "regression"
    return "multiklasse"


def features(slug):
    """Merkmale eines Datensatzes, oder None wenn keine train.csv da ist."""
    train_path = DATA_DIR / slug / "train.csv"
    if not train_path.is_file():
        return None

    train = pd.read_csv(train_path)
    targets = find_targets(train, slug)

    # 'id' ist bei Kaggle durchgaengig nur der Zeilenschluessel.
    ignored = set(targets) | {c for c in train.columns if c.lower() == "id"}
    feature_columns = [c for c in train.columns if c not in ignored]
    feature_frame = train[feature_columns]

    if len(targets) == 1:
        target_series = train[targets[0]]
        ziel_dtype = str(target_series.dtype)
        ziel_n_unique = int(target_series.nunique(dropna=True))
        aufgabe = guess_task(target_series)
    else:
        ziel_dtype = ""
        ziel_n_unique = ""
        aufgabe = "multi_label" if len(targets) > 1 else "unbekannt"

    n_missing = int(train.isna().sum().sum())
    n_cells = train.shape[0] * train.shape[1]

    # Spalten, die fast nur verschiedene Werte enthalten. Das sind haeufig
    # Identifikatoren wie CustomerId oder Surname, die nichts vorhersagen -
    # bei stetigen Groessen kann es aber auch Zufall sein. Darum nur zaehlen,
    # nicht aussortieren.
    n_rows = max(train.shape[0], 1)
    n_nearly_unique = sum(
        1 for c in feature_columns if feature_frame[c].nunique(dropna=True) / n_rows > 0.99
    )

    return {
        "slug": slug,
        "n_zeilen": int(train.shape[0]),
        "n_spalten": int(train.shape[1]),
        "n_features": len(feature_columns),
        "zielspalte": "|".join(targets),
        "ziel_dtype": ziel_dtype,
        "ziel_n_unique": ziel_n_unique,
        "aufgabe_vermutet": aufgabe,
        "n_numerisch": feature_frame.select_dtypes(include="number").shape[1],
        "n_kategorial": feature_frame.select_dtypes(exclude="number").shape[1],
        "n_fast_eindeutig": n_nearly_unique,
        "n_fehlende_werte": n_missing,
        "anteil_fehlend_prozent": round(100 * n_missing / n_cells, 3) if n_cells else 0,
        "spalten_mit_fehlenden": int((train.isna().sum() > 0).sum()),
        "train_mb": round(train_path.stat().st_size / 1_048_576, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", help="nur Slugs mit diesem Teilstring")
    args = parser.parse_args()

    if not DATA_DIR.is_dir():
        raise SystemExit("Kein data/-Ordner. Erst download_competitions.py laufen lassen.")

    slugs = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    if args.only:
        needles = [n.lower() for n in args.only]
        slugs = [s for s in slugs if any(n in s for n in needles)]
    if not slugs:
        raise SystemExit("Keine Datensaetze in " + str(DATA_DIR))

    rows = []
    for i, slug in enumerate(slugs, 1):
        print("[{}/{}] {} ... ".format(i, len(slugs), slug), end="", flush=True)
        try:
            row = features(slug)
        except Exception as exc:
            print(type(exc).__name__ + ": " + str(exc)[:120])
            continue
        if row is None:
            print("keine train.csv")
            continue
        rows.append(row)
        print(
            "{} Zeilen, {} Features, Ziel '{}' ({})".format(
                row["n_zeilen"], row["n_features"], row["zielspalte"], row["aufgabe_vermutet"]
            )
        )

    if not rows:
        raise SystemExit("Nichts auszuwerten.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + str(len(rows)) + " Datensaetze ausgewertet -> " + str(OUT_FILE))


if __name__ == "__main__":
    main()
