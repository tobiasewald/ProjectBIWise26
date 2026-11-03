#!/usr/bin/env python3
"""Die Kaggle-Metriken der Playground-Wettbewerbe als Funktionen.

Jeder Wettbewerb bewertet mit einer eigenen Metrik. Welche das ist, steht als
Freitext in der Spalte 'Metrik' von metadata/playground_overview.csv. Dieses
Modul bildet diesen Text auf einen Code ab und den Code auf eine Funktion.

Ein unbekannter Text fuehrt bewusst zu einem Fehler statt zu einer stillen
Ersatzmetrik - dann hat sich die Uebersichtstabelle geaendert und das muss
auffallen.

Zu jedem Code gehoeren drei Angaben:

    RICHTUNG   +1 wenn ein hoeherer Wert besser ist, -1 wenn ein kleinerer
    BRAUCHT    'proba' wenn Wahrscheinlichkeiten noetig sind, sonst 'werte'
    AUFGABEN   zu welchen Aufgabentypen die Metrik passt

RICHTUNG ist fuer den Rangvergleich noetig: Nur damit laesst sich ueber
Datensaetze mit verschiedenen Metriken hinweg einheitlich ranken.
"""

import re

import numpy as np
from sklearn import metrics as skm

# Freitext aus der Uebersichtstabelle -> Code. Schluessel sind normalisiert
# (klein, ohne Mehrfachleerzeichen), siehe code_fuer_text().
TEXT_ZU_CODE = {
    "auc": "auc",
    "auc (mittelwert ueber 7 targets)": "auc_multilabel",
    "rmse": "rmse",
    "rmsle": "rmsle",
    "accuracy": "accuracy",
    "balanced accuracy": "balanced_accuracy",
    "mcc (matthews correlation coefficient)": "mcc",
    "mape": "mape",
    "r2": "r2",
    "map@3": "map3",
}

RICHTUNG = {
    "auc": +1,
    "auc_multilabel": +1,
    "accuracy": +1,
    "balanced_accuracy": +1,
    "mcc": +1,
    "r2": +1,
    "map3": +1,
    "rmse": -1,
    "rmsle": -1,
    "mape": -1,
}

BRAUCHT = {
    "auc": "proba",
    "auc_multilabel": "proba",
    "map3": "proba",
    "accuracy": "werte",
    "balanced_accuracy": "werte",
    "mcc": "werte",
    "rmse": "werte",
    "rmsle": "werte",
    "mape": "werte",
    "r2": "werte",
}

AUFGABEN = {
    "auc": {"binaer"},
    "auc_multilabel": {"multi_label"},
    "accuracy": {"binaer", "multiklasse"},
    "balanced_accuracy": {"binaer", "multiklasse"},
    "mcc": {"binaer", "multiklasse"},
    "map3": {"multiklasse"},
    "rmse": {"regression"},
    "rmsle": {"regression"},
    "mape": {"regression"},
    "r2": {"regression"},
}


def normalisiere(text):
    """Vereinheitlicht den Freitext, damit Schreibweisen nicht stoeren.

    'R²' und 'R^2' werden zu 'r2', Umlaute zu Ersatzschreibweisen - sonst
    haengt die Zuordnung daran, wie die Tabelle kodiert gespeichert wurde.
    """
    t = text.strip().lower()
    for alt, neu in (
        ("²", "2"), ("^2", "2"),
        ("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss"),
    ):
        t = t.replace(alt, neu)
    return re.sub(r"\s+", " ", t)


def code_fuer_text(text):
    """Code zur Metrikangabe aus der Uebersichtstabelle."""
    schluessel = normalisiere(text)
    if schluessel not in TEXT_ZU_CODE:
        bekannt = ", ".join(sorted(TEXT_ZU_CODE))
        raise KeyError(
            "Unbekannte Metrik {!r} (normalisiert {!r}). Bekannt: {}".format(
                text, schluessel, bekannt
            )
        )
    return TEXT_ZU_CODE[schluessel]


def _map_at_k(y, proba, klassen, k=3):
    """Mean Average Precision @ k bei genau einer richtigen Klasse pro Zeile.

    Dann ist die Average Precision einer Zeile 1/Rang, wenn die richtige
    Klasse unter den ersten k liegt, sonst 0.
    """
    k = min(k, proba.shape[1])
    beste = np.argsort(-proba, axis=1)[:, :k]
    getroffen = np.asarray(klassen)[beste] == np.asarray(y)[:, None]
    gewichte = 1.0 / np.arange(1, k + 1)
    return float((getroffen * gewichte).sum(axis=1).mean())


def bewerte(code, y_wahr, vorhersage, klassen=None):
    """Metrikwert fuer einen Fold.

    vorhersage ist je nach BRAUCHT[code] ein Wahrscheinlichkeitsarray
    (n x Klassen) oder ein Vektor aus Labels bzw. Zahlen.
    """
    y = np.asarray(y_wahr)
    p = np.asarray(vorhersage)

    if code == "auc":
        # Bei zwei Klassen zaehlt nur die Spalte der positiven Klasse.
        return float(skm.roc_auc_score(y, p[:, 1] if p.ndim == 2 else p))
    if code == "auc_multilabel":
        # Ein AUC pro Zielspalte, dann der Mittelwert - so rechnet Kaggle
        # bei den Multi-Label-Episoden.
        return float(skm.roc_auc_score(y, p, average="macro"))
    if code == "map3":
        return _map_at_k(y, p, klassen)
    if code == "accuracy":
        return float(skm.accuracy_score(y, p))
    if code == "balanced_accuracy":
        return float(skm.balanced_accuracy_score(y, p))
    if code == "mcc":
        return float(skm.matthews_corrcoef(y, p))
    if code == "rmse":
        return float(skm.root_mean_squared_error(y, p))
    if code == "rmsle":
        # Der Logarithmus verbietet negative Werte. Ein ungetuntes Modell
        # sagt die durchaus vorher, deshalb wird abgeschnitten.
        return float(skm.root_mean_squared_log_error(y, np.clip(p, 0, None)))
    if code == "mape":
        return float(skm.mean_absolute_percentage_error(y, p))
    if code == "r2":
        return float(skm.r2_score(y, p))
    raise KeyError("Keine Funktion fuer Code " + repr(code))


def passt_zur_aufgabe(code, aufgabe):
    """Ob Metrik und erkannter Aufgabentyp zusammengehen."""
    return aufgabe in AUFGABEN.get(code, set())
