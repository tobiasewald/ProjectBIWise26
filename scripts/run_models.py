#!/usr/bin/env python3
"""Laesst die Modelle ungetunt ueber die Datensaetze laufen.

Random Forest und XGBoost (--modelle rf xgb) sowie die Fundamentmodelle
TabPFN und TabFM (--modelle tabpfn tabfm). Pro Datensatz wird mit der Metrik
bewertet, die Kaggle fuer diesen Wettbewerb vorgibt (Spalte 'Metrik' der
Uebersichtstabelle). Bewertet wird per Kreuzvalidierung auf train.csv, weil
die Testlabels bei Kaggle nicht oeffentlich sind.

Zwei Varianten:

    n10000   Datensatz auf 10.000 Zeilen verkleinert. Hauptauswertung.
    voll     alle Zeilen. Nur fuer Wald und Boosting sinnvoll und zeigt,
             was die Deckelung auf 10.000 Zeilen kostet.

Die Deckelung ist keine technische Grenze, sondern eine methodische
Entscheidung: Alle Modelle sollen dieselbe Datenmenge sehen. Die
Fundamentmodelle lesen ihre Trainingsdaten als Kontext statt daraus zu
lernen und sind darin begrenzt - TabFM etwa voreingestellt bei 100 Zeilen.

Die Vorverarbeitung ist fuer alle Modelle gleich (Median fuer Zahlen,
Ordinalkodierung fuer Text) und wird je Fold nur auf den Trainingsdaten
angepasst. Sonst vergleicht man Vorverarbeitungen statt Modelle.

Jede fertige Kombination wird sofort angehaengt und beim naechsten Lauf
uebersprungen - eine abgebrochene Colab-Sitzung kostet also nur den
laufenden Datensatz.

    python scripts/run_models.py --only s5e3
    python scripts/run_models.py
    python scripts/run_models.py --modelle tabpfn --laden
    python scripts/run_models.py --variante voll --modelle rf
"""

import argparse
import csv
import os
import platform
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

import metrics as mk

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OVERVIEW = ROOT / "metadata" / "playground_overview.csv"
FEATURES = ROOT / "metadata" / "dataset_features.csv"
OUT_FILE = ROOT / "results" / "model_results.csv"

SEED = 42
FOLDS = 5
SUBSAMPLE = 10_000

COLUMNS = [
    "slug", "modell", "variante", "aufgabe", "metrik", "metrik_code",
    "richtung", "n_zeilen_genutzt", "n_features", "folds",
    "wert_mittel", "wert_std", "werte_folds",
    "sekunden_fit", "sekunden_gesamt", "status", "hinweis",
    "modell_version", "umgebung", "zeitpunkt",
]

KLASSIFIKATION = {"binaer", "multiklasse"}
MODELLE = ["rf", "xgb", "tabpfn", "tabfm"]

# TabPFN-Gewichtsstand. Gerechnet wird in jedem Fall lokal; der Unterschied
# liegt nur darin, ob die Gewichte frei heruntergeladen werden duerfen.
#
#   V2     Prior Labs License (Apache 2.0 mit Namensnennung), laedt ohne Konto.
#   V3_5   aktueller Stand. Prior Labs verlangt einmal, dass die Lizenz
#          tabpfn-3-5-license-v1.0 im Konto bestaetigt wird; geprueft wird das
#          ueber die Umgebungsvariable TABPFN_TOKEN (API-Key aus
#          https://ux.priorlabs.ai/account). Die Gewichtsdatei selbst liegt
#          offen auf Hugging Face, ein Konto dort ist nicht noetig.
#
# V3_5 ist der Standard, weil die Checkpoint-Datei ein Datum im Namen traegt
# (tabpfn-v3.5-20260909.safetensors) und der gemessene Stand damit zitierbar
# ist. Umstellbar ueber --tabpfn-version.
TABPFN_VERSION = "V3_5"

# TabFM laedt seine Gewichte von Hugging Face. Das darf nicht pro Fold neu
# passieren, deshalb einmal pro Aufgabenart hier merken.
_TABFM_GEWICHTE = {}


def _mit_startwert(bauart, **kwargs):
    """Setzt random_state, wenn das Modell den Parameter kennt.

    Die Fundamentmodelle ziehen Stichproben aus den Kontextzeilen. Ohne
    Startwert waere ein zweiter Lauf nicht derselbe.
    """
    try:
        return bauart(random_state=SEED, **kwargs)
    except TypeError:
        return bauart(**kwargs)


def modell_version(name):
    """Welcher Gewichtsstand gemessen wurde.

    Bei den Fundamentmodellen entscheidet das ueber das Ergebnis, bei rf und
    xgb steht die Bibliotheksversion schon in umgebung().
    """
    if name == "tabpfn":
        import tabpfn
        return "tabpfn {} {}".format(getattr(tabpfn, "__version__", "?"), TABPFN_VERSION)
    if name == "tabfm":
        return "tabfm v1.0.0"
    return ""


def umgebung():
    """Wo und womit gerechnet wurde.

    Steht in jeder Ergebniszeile, damit nicht unbemerkt Messungen von
    verschiedenen Rechnern in einer Tabelle landen. Die Metrikwerte sind bei
    festem Startwert gleich, die Zeiten nicht - und die Voreinstellungen von
    scikit-learn und XGBoost koennen sich zwischen Versionen aendern.
    """
    try:
        import google.colab  # noqa: F401
        ort = "colab"
    except ImportError:
        ort = platform.system().lower()
    return "{} py{} skl{} xgb{}".format(
        ort, platform.python_version(), sklearn.__version__, xgboost.__version__
    )


def modell(name, aufgabe):
    """Ungetuntes Modell. Gesetzt werden nur Startwert und Kernzahl."""
    if name == "rf":
        bauart = RandomForestRegressor if aufgabe == "regression" else RandomForestClassifier
        return bauart(random_state=SEED, n_jobs=-1)
    if name == "xgb":
        bauart = XGBRegressor if aufgabe == "regression" else XGBClassifier
        return bauart(random_state=SEED, n_jobs=-1, verbosity=0)

    # Die beiden Fundamentmodelle werden erst hier importiert, damit ein Lauf
    # mit nur rf und xgb ohne torch und ohne Gewichte funktioniert.
    if name == "tabpfn":
        # Ohne gueltigen TABPFN_TOKEN wuerde tabpfn versuchen, einen Browser
        # zu oeffnen, und auf einen Rueckruf an localhost warten. In einem
        # Stapellauf - und in Colab, wo localhost nicht der eigene Rechner
        # ist - haengt das nur. Mit dieser Variable bricht es stattdessen
        # sofort mit einer Anleitung ab.
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")

        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        bauart = TabPFNRegressor if aufgabe == "regression" else TabPFNClassifier
        geschaetzt = bauart.create_default_for_version(ModelVersion[TABPFN_VERSION])
        try:
            geschaetzt.set_params(random_state=SEED)
        except (ValueError, AttributeError):
            pass
        return geschaetzt

    if name == "tabfm":
        from tabfm import TabFMClassifier, TabFMRegressor
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

        # TabFM waehlt das Rechengeraet nicht selbst. load() hat
        # device=None voreingestellt, und das heisst laut Quelltext CPU -
        # anders als TabPFN, das eine vorhandene CUDA-Karte von sich aus
        # nimmt. Ohne diese Zeilen laeuft das Modell also auf der CPU,
        # obendrein in bfloat16, das dort nur nachgebildet wird. Ein
        # einzelner Datensatz braucht so Minuten statt Sekunden.
        import torch

        geraet = "cuda" if torch.cuda.is_available() else "cpu"

        art = "regression" if aufgabe == "regression" else "classification"
        if art not in _TABFM_GEWICHTE:
            print("TabFM laedt Gewichte ({}) auf {} ... ".format(art, geraet),
                  end="", flush=True)
            _TABFM_GEWICHTE[art] = tabfm_v1_0_0.load(
                model_type=art, device=geraet
            )
        bauart = TabFMRegressor if art == "regression" else TabFMClassifier
        return _mit_startwert(bauart, model=_TABFM_GEWICHTE[art])

    raise ValueError("Unbekanntes Modell " + repr(name))


def read_overview():
    """slug -> Metrikangabe aus der Uebersichtstabelle."""
    aus = {}
    with open(OVERVIEW, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            slug = (row.get("slugs") or "").strip()
            if re.fullmatch(r"[a-z0-9-]+", slug):
                aus[slug] = (row.get("Metrik") or "").strip()
    return aus


def read_features():
    """slug -> Zeile der Merkmalstabelle (Aufgabentyp, Zielspalte)."""
    with open(FEATURES, newline="", encoding="utf-8-sig") as handle:
        return {r["slug"]: r for r in csv.DictReader(handle, delimiter=";")}


def read_done():
    """Erfolgreich gerechnete Kombinationen (slug, modell, variante).

    Nur status=ok zaehlt als fertig. Eine Fehlerzeile ist kein Ergebnis: Wer
    einen Lauf ohne TABPFN_TOKEN startet, bekommt fuer jeden Datensatz eine
    Zeile mit TabPFNLicenseError - wuerden die als fertig gelten, liefe der
    naechste Lauf nach dem Eintragen des Tokens komplett ins Leere und man
    haette am Ende eine Tabelle ohne TabPFN, ohne es zu merken.

    Die Fehlerzeilen bleiben in der Datei stehen; die Auswertung filtert sie.
    """
    if not OUT_FILE.is_file():
        return set()
    with open(OUT_FILE, newline="", encoding="utf-8-sig") as handle:
        return {
            (r["slug"], r["modell"], r["variante"])
            for r in csv.DictReader(handle, delimiter=";")
            if r.get("status") == "ok"
        }


def append_row(row):
    """Haengt eine Ergebniszeile an - sofort, nicht am Ende des Laufs."""
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    neu = not OUT_FILE.is_file()
    with open(OUT_FILE, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter=";", extrasaction="ignore")
        if neu:
            writer.writeheader()
        writer.writerow(row)


def vorverarbeitung(X):
    """Gleiche Vorverarbeitung fuer alle Modelle, je Fold neu angepasst.

    Zahlen werden mit dem Median gefuellt. Textspalten sind vorher auf str
    gegossen, fehlende Werte sind damit die eigene Kategorie 'nan'.
    """
    zahlen = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    text = [c for c in X.columns if c not in zahlen]
    return ColumnTransformer([
        ("zahl", SimpleImputer(strategy="median"), zahlen),
        ("text", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), text),
    ], remainder="drop")


def lade(slug, zielspalten):
    """train.csv als X und y, ohne id-Spalte und ohne Zeilen ohne Ziel."""
    train = pd.read_csv(DATA_DIR / slug / "train.csv")
    train = train.dropna(subset=zielspalten)

    weg = set(zielspalten) | {c for c in train.columns if c.lower() == "id"}
    X = train[[c for c in train.columns if c not in weg]].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            # Gemischte Typen in einer Objektspalte brechen die Kodierung,
            # deshalb durchgaengig str. NaN wird dabei zu 'nan'.
            X[c] = X[c].astype(str)

    y = train[zielspalten[0]] if len(zielspalten) == 1 else train[zielspalten]
    return X, y


def verkleinern(X, y, aufgabe):
    """Auf SUBSAMPLE Zeilen ziehen, bei Klassifikation schichtweise.

    Fester Startwert, damit alle Modelle genau dieselben Zeilen sehen.
    """
    if len(X) <= SUBSAMPLE:
        return X, y
    schichten = y if aufgabe in KLASSIFIKATION else None
    X_klein, _, y_klein, _ = train_test_split(
        X, y, train_size=SUBSAMPLE, random_state=SEED, stratify=schichten
    )
    return X_klein, y_klein


def folds_fuer(aufgabe, y):
    """Dieselbe Aufteilung fuer alle Modelle - gepaarter Vergleich."""
    if aufgabe in KLASSIFIKATION:
        return StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(y, y)
    return KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(y)


def ein_fold_multilabel(name, X_tr, X_te, y_tr, y_te, vorbereiter):
    """Ein Modell pro Zielspalte, Wahrscheinlichkeiten als Matrix zurueck.

    Genau so bewertet Kaggle die Multi-Label-Episoden: ein AUC je Zielspalte,
    danach der Mittelwert.
    """
    spalten, wahr, proba, sekunden = [], [], [], 0.0
    for ziel in y_tr.columns:
        if y_tr[ziel].nunique() < 2:
            continue  # ohne beide Klassen im Fold ist kein AUC definiert
        pipe = Pipeline([("vor", vorbereiter), ("modell", modell(name, "binaer"))])
        start = time.perf_counter()
        pipe.fit(X_tr, y_tr[ziel])
        sekunden += time.perf_counter() - start
        proba.append(pipe.predict_proba(X_te)[:, 1])
        wahr.append(y_te[ziel].to_numpy())
        spalten.append(ziel)
    return np.column_stack(wahr), np.column_stack(proba), sekunden, spalten


def ein_fold(name, aufgabe, code, X_tr, X_te, y_tr, y_te, vorbereiter):
    """Trainiert, sagt vorher, gibt Metrikwert und Trainingszeit zurueck."""
    if aufgabe == "multi_label":
        wahr, proba, sekunden, _ = ein_fold_multilabel(
            name, X_tr, X_te, y_tr, y_te, vorbereiter
        )
        return mk.bewerte(code, wahr, proba), sekunden

    pipe = Pipeline([("vor", vorbereiter), ("modell", modell(name, aufgabe))])
    start = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    sekunden = time.perf_counter() - start

    if mk.BRAUCHT[code] == "proba":
        vorhersage = pipe.predict_proba(X_te)
        klassen = pipe.named_steps["modell"].classes_
    else:
        vorhersage = pipe.predict(X_te)
        klassen = None
    return mk.bewerte(code, y_te, vorhersage, klassen), sekunden


def rechne(slug, name, variante, zeile, metrik_text):
    """Eine Kombination aus Datensatz, Modell und Variante."""
    aufgabe = zeile["aufgabe_vermutet"]
    zielspalten = [z for z in zeile["zielspalte"].split("|") if z]
    code = mk.code_fuer_text(metrik_text)

    basis = {
        "slug": slug, "modell": name, "variante": variante, "aufgabe": aufgabe,
        "metrik": metrik_text, "metrik_code": code, "richtung": mk.RICHTUNG[code],
        "folds": FOLDS, "umgebung": umgebung(), "modell_version": modell_version(name),
        "zeitpunkt": datetime.now().isoformat(timespec="seconds"),
        "n_zeilen_genutzt": "", "n_features": "", "wert_mittel": "", "wert_std": "",
        "werte_folds": "", "sekunden_fit": "", "sekunden_gesamt": "", "hinweis": "",
    }

    if not mk.passt_zur_aufgabe(code, aufgabe):
        basis["status"] = "metrik_passt_nicht"
        basis["hinweis"] = "{} ist fuer {} nicht definiert".format(code, aufgabe)
        return basis

    gesamt_start = time.perf_counter()
    X, y = lade(slug, zielspalten)
    if variante == "n10000":
        X, y = verkleinern(X, y, aufgabe)

    basis["n_zeilen_genutzt"] = len(X)
    basis["n_features"] = X.shape[1]

    hinweise = []
    if code == "rmsle" and (np.asarray(y) < 0).any():
        basis["status"] = "metrik_unmoeglich"
        basis["hinweis"] = "negative Zielwerte, RMSLE nicht definiert"
        return basis
    if code == "mape" and (np.asarray(y) == 0).any():
        hinweise.append("Nullwerte im Ziel, MAPE verzerrt")

    # XGBoost erwartet Klassen als 0..n-1, der Wald nicht. Einmal kodieren
    # und beide Modelle bekommen dasselbe y.
    if aufgabe in KLASSIFIKATION:
        y = pd.Series(LabelEncoder().fit_transform(y), index=X.index)

    werte, sekunden_fit = [], 0.0
    for i_tr, i_te in folds_fuer(aufgabe, y):
        wert, sek = ein_fold(
            name, aufgabe, code,
            X.iloc[i_tr], X.iloc[i_te], y.iloc[i_tr], y.iloc[i_te],
            vorverarbeitung(X),
        )
        werte.append(wert)
        sekunden_fit += sek

    basis.update({
        "wert_mittel": round(float(np.mean(werte)), 6),
        "wert_std": round(float(np.std(werte, ddof=1)), 6),
        "werte_folds": "|".join("{:.6f}".format(w) for w in werte),
        "sekunden_fit": round(sekunden_fit, 2),
        "sekunden_gesamt": round(time.perf_counter() - gesamt_start, 2),
        "status": "ok",
        "hinweis": "; ".join(hinweise),
    })
    return basis


def hole_daten(slug):
    """Laedt den Datensatz von Kaggle, wenn er lokal fehlt.

    Fuer Colab: Dort ist die Platte bei jeder Sitzung leer, und von Kaggle in
    ein Google-Rechenzentrum zu laden ist schneller, als 3,7 GB durch eine
    private Leitung zu schieben. Importiert wird erst hier, damit der Lauf auf
    vorhandenen Daten ohne requests und ohne Token funktioniert.
    """
    import download_competitions as dl

    dl.DATA_DIR = DATA_DIR
    status, sekunden, megabytes = dl.download(slug, dl.read_token())
    if status not in ("ok", "uebersprungen"):
        raise RuntimeError("Download von {} fehlgeschlagen: {}".format(slug, status))
    return sekunden, megabytes


def main():
    global DATA_DIR, OUT_FILE, TABPFN_VERSION

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", help="nur Slugs mit diesem Teilstring")
    parser.add_argument("--modelle", nargs="+", default=["rf", "xgb"], choices=MODELLE)
    parser.add_argument("--variante", default="n10000", choices=["n10000", "voll"])
    parser.add_argument("--neu", action="store_true", help="fertige Kombinationen neu rechnen")
    parser.add_argument("--ausgabe", help="Pfad der Ergebnis-CSV (z. B. ein Ordner auf Drive)")
    parser.add_argument("--daten", help="Ordner mit den Datensaetzen")
    parser.add_argument("--laden", action="store_true",
                        help="fehlende Datensaetze von Kaggle holen (fuer Colab)")
    parser.add_argument("--daten-loeschen", action="store_true",
                        help="Datensatz nach der Rechnung wieder entfernen, spart Platte")
    parser.add_argument("--tabpfn-version", default=TABPFN_VERSION,
                        help="Gewichtsstand von TabPFN, z. B. V2 oder V3_5")
    args = parser.parse_args()

    TABPFN_VERSION = args.tabpfn_version
    if args.daten:
        DATA_DIR = Path(args.daten)
    if args.ausgabe:
        OUT_FILE = Path(args.ausgabe)
        if OUT_FILE.is_dir():
            OUT_FILE = OUT_FILE / "model_results.csv"

    overview = read_overview()
    features = read_features()
    fertig = set() if args.neu else read_done()

    slugs = sorted(features)
    if args.only:
        nadeln = [n.lower() for n in args.only]
        slugs = [s for s in slugs if any(n in s for n in nadeln)]
    if not slugs:
        raise SystemExit("Keine passenden Datensaetze.")

    # Klein nach gross, damit bei einem Abbruch schon Ergebnisse da sind.
    slugs.sort(key=lambda s: int(features[s]["n_zeilen"]))

    aufgaben = [(s, m) for s in slugs for m in args.modelle]
    for i, (slug, name) in enumerate(aufgaben, 1):
        kopf = "[{}/{}] {} {} {}".format(i, len(aufgaben), slug, name, args.variante)
        if (slug, name, args.variante) in fertig:
            print(kopf + " ... fertig, uebersprungen")
            continue
        if not (DATA_DIR / slug / "train.csv").is_file():
            if not args.laden:
                print(kopf + " ... keine train.csv")
                continue
            print(kopf + " ... lade ... ", end="", flush=True)
            try:
                hole_daten(slug)
            except Exception as exc:
                print("Download fehlgeschlagen: " + str(exc)[:90])
                continue
        else:
            print(kopf + " ... ", end="", flush=True)
        try:
            row = rechne(slug, name, args.variante, features[slug], overview[slug])
        except Exception as exc:
            row = {
                "slug": slug, "modell": name, "variante": args.variante,
                "status": "fehler", "hinweis": type(exc).__name__ + ": " + str(exc)[:200],
                "umgebung": umgebung(),
                "zeitpunkt": datetime.now().isoformat(timespec="seconds"),
            }
        append_row(row)

        if row["status"] == "ok":
            print("{} {:.5f} (+-{:.5f}) in {:.1f} s".format(
                row["metrik_code"], row["wert_mittel"], row["wert_std"], row["sekunden_fit"]))
        else:
            print(row["status"] + ": " + str(row.get("hinweis", ""))[:90])

        # Nur loeschen, wenn der Datensatz auch wieder geholt werden kann,
        # sonst entfernt man Daten, die niemand neu beschafft.
        if args.daten_loeschen and args.laden and name == args.modelle[-1]:
            shutil.rmtree(DATA_DIR / slug, ignore_errors=True)

    print("\nErgebnisse -> " + str(OUT_FILE))


if __name__ == "__main__":
    main()
