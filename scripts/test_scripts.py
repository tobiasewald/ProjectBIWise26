#!/usr/bin/env python3
"""Testet die beiden Skripte gegen einen lokalen HTTP-Server.

Kaggle wird nicht kontaktiert, ein Token ist nicht noetig.

    python scripts/test_scripts.py
"""

import csv
import http.server
import io
import os
import shutil
import socketserver
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download_competitions as dl  # noqa: E402
import extract_features as ef  # noqa: E402

TOKEN = "KGAT_" + "t" * 30
WORK = Path(tempfile.mkdtemp(prefix="kpg_test_"))
failures = []


def check(label, ok, detail=""):
    print("  {} {}{}".format("OK    " if ok else "FEHLER", label, "  " + detail if detail else ""))
    if not ok:
        failures.append(label)


def make_zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


ARCHIVES = {
    "gut": make_zip({
        "train.csv": "id,alter,stadt,preis\n1,30,BER,100\n2,40,HH,200\n3,50,BER,300\n",
        "test.csv": "id,alter,stadt\n4,60,HH\n",
    }),
    "boese": make_zip({"../../pwned.csv": "boom\n"}),
}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        slug = self.path.rsplit("/", 1)[-1]
        if self.headers.get("Authorization") != "Bearer " + TOKEN:
            body, code = b'{"code":401}', 401
        elif slug in ARCHIVES:
            body, code = ARCHIVES[slug], 200
        elif slug == "ohne-regeln":
            body, code = b'{"code":403,"message":"must accept rules"}', 403
        else:
            body, code = b'{"code":404}', 404
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

dl.API = "http://127.0.0.1:{}/".format(server.server_address[1])
dl.DATA_DIR = WORK / "data"
headers = {"Authorization": "Bearer " + TOKEN}

print("\n=== Download ===")
status, seconds, megabytes = dl.download("gut", headers)
check("status ok", status == "ok", status)
check("Zeit gemessen", seconds > 0, "{:.4f} s".format(seconds))
check("Groesse gemessen", megabytes > 0)
check("train.csv entpackt", (dl.DATA_DIR / "gut" / "train.csv").is_file())
check("Archiv geloescht", not (dl.DATA_DIR / "gut" / "archive.zip").exists())

print("\n=== Zweiter Lauf ueberspringt ===")
status, _, _ = dl.download("gut", headers)
check("status uebersprungen", status == "uebersprungen", status)

print("\n=== Fehlerfaelle ===")
status, _, _ = dl.download("ohne-regeln", headers)
check("403 -> regeln_fehlen", status == "regeln_fehlen", status)
status, _, _ = dl.download("gibt-es-nicht", headers)
check("404 -> http_404", status == "http_404", status)
(dl.DATA_DIR / "gut" / ".done").unlink()  # sonst greift die Wiederaufnahme
status, _, _ = dl.download("gut", {"Authorization": "Bearer falsch"})
check("falsches Token -> http_401", status == "http_401", status)

try:
    dl.download("boese", headers)
    check("Zip-Slip wird abgewiesen", False, "kein Fehler geworfen")
except ValueError as exc:
    check("Zip-Slip wird abgewiesen", "Unsicher" in str(exc), str(exc)[:60])
check("nichts ausserhalb entpackt", not (WORK / "pwned.csv").exists())

print("\n=== Token lesen ===")
saved = {k: os.environ.get(k) for k in ("KAGGLE_API_TOKEN",)}
# Auf echte Tokendateien des Nutzers darf der Test nicht zugreifen.
dl.PROJECT_TOKEN_FILE = WORK / "projekt_token"
dl.TOKEN_FILE = WORK / "home_token"
try:
    os.environ["KAGGLE_API_TOKEN"] = TOKEN
    check("Header aus Umgebung", dl.read_token() == headers)

    os.environ["KAGGLE_API_TOKEN"] = dl.PLACEHOLDER
    try:
        dl.read_token()
        check("Platzhalter wird erkannt", False, "kein Fehler")
    except SystemExit as exc:
        check("Platzhalter wird erkannt", "Platzhalter" in str(exc), str(exc)[:50])

    os.environ["KAGGLE_API_TOKEN"] = "zu-kurz"
    try:
        dl.read_token()
        check("unplausibles Token wird erkannt", False, "kein Fehler")
    except SystemExit as exc:
        check("unplausibles Token wird erkannt", "KGAT_" in str(exc))

    os.environ.pop("KAGGLE_API_TOKEN")
    dl.TOKEN_FILE.write_text(TOKEN + "\n", encoding="utf-8")
    check("Header aus Home-Datei, Umbruch entfernt", dl.read_token() == headers)

    dl.PROJECT_TOKEN_FILE.write_text("KGAT_" + "p" * 30, encoding="utf-8")
    check(
        "Projektordner hat Vorrang vor Home",
        dl.read_token()["Authorization"].endswith("p" * 30),
    )
finally:
    for key, value in saved.items():
        os.environ.pop(key, None)
        if value is not None:
            os.environ[key] = value

print("\n=== Slugs aus der Tabelle ===")
slugs = dl.read_slugs()
check("83 Slugs", len(slugs) == 83, str(len(slugs)))
check("keine Platzhalterzeilen", all(s and "-" in s for s in slugs))

print("\n=== Merkmale extrahieren ===")
ef.DATA_DIR = dl.DATA_DIR
row = ef.features("gut")
check("3 Zeilen", row["n_zeilen"] == 3, str(row["n_zeilen"]))
check("4 Spalten", row["n_spalten"] == 4, str(row["n_spalten"]))
check("2 Features (id und Ziel raus)", row["n_features"] == 2, str(row["n_features"]))
check("Ziel aus train/test-Differenz", row["zielspalte"] == "preis", row["zielspalte"])
check("1 numerisches Feature", row["n_numerisch"] == 1, str(row["n_numerisch"]))
check("1 kategoriales Feature", row["n_kategorial"] == 1, str(row["n_kategorial"]))
check("keine fehlenden Werte", row["n_fehlende_werte"] == 0)

(dl.DATA_DIR / "luecken").mkdir(parents=True, exist_ok=True)
(dl.DATA_DIR / "luecken" / "train.csv").write_text(
    "id,a,b,ziel\n1,1,x,0\n2,,y,1\n3,3,,0\n", encoding="utf-8"
)
(dl.DATA_DIR / "luecken" / "test.csv").write_text("id,a,b\n4,4,z\n", encoding="utf-8")
row = ef.features("luecken")
check("2 fehlende Werte gezaehlt", row["n_fehlende_werte"] == 2, str(row["n_fehlende_werte"]))
check("2 Spalten mit Luecken", row["spalten_mit_fehlenden"] == 2, str(row["spalten_mit_fehlenden"]))
check("binaeres Ziel erkannt", row["aufgabe_vermutet"] == "binaer", row["aufgabe_vermutet"])
check("fehlt-Anteil berechnet", row["anteil_fehlend_prozent"] > 0, str(row["anteil_fehlend_prozent"]))

check("ohne train.csv -> None", ef.features("gibt-es-nicht") is None)

print("\n=== Lauf mit --only loescht die anderen Zeilen nicht ===")
# Unbedingt umbiegen, sonst schreibt der Test in die echte Merkmalstabelle.
ef.OUT_FILE = WORK / "features.csv"


def slugs_in_tabelle():
    with open(ef.OUT_FILE, newline="", encoding="utf-8-sig") as handle:
        return {r["slug"] for r in csv.DictReader(handle, delimiter=";")}


argv_vorher = sys.argv
try:
    sys.argv = ["extract_features.py"]
    ef.main()
    check("erst beide Datensaetze drin", slugs_in_tabelle() == {"gut", "luecken"},
          str(sorted(slugs_in_tabelle())))

    sys.argv = ["extract_features.py", "--only", "gut"]
    ef.main()
    check("nach --only immer noch beide", slugs_in_tabelle() == {"gut", "luecken"},
          str(sorted(slugs_in_tabelle())))
finally:
    sys.argv = argv_vorher

check("n_fast_eindeutig ist weg", "n_fast_eindeutig" not in ef.COLUMNS)

print("\n=== Metriken: Zuordnung aus der Uebersichtstabelle ===")
import metrics as mk  # noqa: E402
import run_models as rm  # noqa: E402

check("alle Codes haben eine Richtung", set(mk.RICHTUNG) == set(mk.TEXT_ZU_CODE.values()))
check("alle Codes sagen was sie brauchen", set(mk.BRAUCHT) == set(mk.RICHTUNG))
check("alle Codes nennen ihre Aufgaben", set(mk.AUFGABEN) == set(mk.RICHTUNG))

# Jede Metrikangabe der echten Tabelle muss zuzuordnen sein, sonst rechnet
# der Durchlauf spaeter fuer einen Datensatz die falsche Zahl.
overview = rm.read_overview()
geladen = rm.read_features()
unbekannt = []
for slug in geladen:
    try:
        mk.code_fuer_text(overview[slug])
    except KeyError:
        unbekannt.append(overview[slug])
check("alle {} Metrikangaben zugeordnet".format(len(geladen)), not unbekannt, str(unbekannt))

check("R2 mit Hochzahl erkannt", mk.code_fuer_text("R\u00b2") == "r2")
check("R^2 erkannt", mk.code_fuer_text("R^2") == "r2")
check("Gross-/Kleinschreibung egal", mk.code_fuer_text("  auc  ") == "auc")
try:
    mk.code_fuer_text("F1 Score")
    check("unbekannte Metrik wirft Fehler", False, "kein Fehler")
except KeyError as exc:
    check("unbekannte Metrik wirft Fehler", "Unbekannte Metrik" in str(exc))

print("\n=== Metriken: Rechnung ===")
import numpy as np_  # noqa: E402

# Richtige Klasse steht in Zeile 1 auf Rang 1, in Zeile 2 auf Rang 2,
# in Zeile 3 auf Rang 4 - also ausserhalb der ersten drei.
proba = np_.array([
    [0.7, 0.2, 0.1, 0.0],
    [0.2, 0.7, 0.1, 0.0],
    [0.4, 0.3, 0.2, 0.1],
])
erwartet = (1.0 + 0.5 + 0.0) / 3
wert = mk.bewerte("map3", [0, 0, 3], proba, klassen=[0, 1, 2, 3])
check("MAP@3 rechnet 1/Rang", abs(wert - erwartet) < 1e-9, "{:.4f}".format(wert))

check("AUC nimmt die positive Spalte",
      mk.bewerte("auc", [0, 0, 1, 1], np_.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])) == 1.0)
check("RMSE bei perfekter Vorhersage 0", mk.bewerte("rmse", [1.0, 2.0], [1.0, 2.0]) == 0.0)
# Ein ungetuntes Regressionsmodell sagt negative Werte vorher, RMSLE
# vertraegt die nicht - abgeschnitten darf es nicht abstuerzen.
check("RMSLE schneidet negative Vorhersagen ab",
      mk.bewerte("rmsle", [0.0, 1.0], [-5.0, 1.0]) == 0.0)
check("Metrik passt nicht zur Aufgabe", not mk.passt_zur_aufgabe("rmse", "binaer"))
check("Metrik passt zur Aufgabe", mk.passt_zur_aufgabe("auc", "binaer"))

print("\n=== Durchlauf: Teilstichprobe ===")
import pandas as pd_  # noqa: E402

X_test = pd_.DataFrame({"a": range(100)})
y_test = pd_.Series([0] * 90 + [1] * 10)
rm.SUBSAMPLE = 20
k1 = rm.verkleinern(X_test, y_test, "binaer")
k2 = rm.verkleinern(X_test, y_test, "binaer")
check("genau SUBSAMPLE Zeilen", len(k1[0]) == 20, str(len(k1[0])))
check("zweimal dieselben Zeilen", list(k1[0].index) == list(k2[0].index))
check("Schichtung erhaelt das Verhaeltnis", k1[1].sum() == 2, str(k1[1].sum()))
check("kleiner Datensatz bleibt ganz", len(rm.verkleinern(X_test.head(5), y_test.head(5), "binaer")[0]) == 5)

print("\n=== Durchlauf: ganze Kette auf Minidaten ===")
mini = WORK / "mdata" / "mini"
mini.mkdir(parents=True, exist_ok=True)
rows_mini = ["id,zahl,text,ziel"]
for i in range(60):
    ziel = i % 2
    zahl = "" if i == 5 else str(ziel * 10 + i % 3)  # eine Luecke fuer den Imputer
    rows_mini.append("{},{},{},{}".format(i, zahl, "a" if i % 3 else "b", ziel))
(mini / "train.csv").write_text("\n".join(rows_mini) + "\n", encoding="utf-8")

rm.DATA_DIR = WORK / "mdata"
rm.OUT_FILE = WORK / "ergebnisse.csv"
rm.FOLDS = 3
rm.SUBSAMPLE = 10_000
zeile_mini = {"aufgabe_vermutet": "binaer", "zielspalte": "ziel", "n_zeilen": "60"}

ergebnis = rm.rechne("mini", "rf", "n10000", zeile_mini, "AUC")
check("Status ok", ergebnis["status"] == "ok", str(ergebnis.get("hinweis")))
check("id wird nicht als Feature gezaehlt", ergebnis["n_features"] == 2, str(ergebnis["n_features"]))
check("Luecke bricht nichts", ergebnis["n_zeilen_genutzt"] == 60, str(ergebnis["n_zeilen_genutzt"]))
check("3 Foldwerte", len(ergebnis["werte_folds"].split("|")) == 3, ergebnis["werte_folds"])
check("AUC ueber Zufall", ergebnis["wert_mittel"] > 0.7, str(ergebnis["wert_mittel"]))
check("Richtung mitgeschrieben", ergebnis["richtung"] == 1)

ergebnis_xgb = rm.rechne("mini", "xgb", "n10000", zeile_mini, "AUC")
check("XGBoost laeuft auch", ergebnis_xgb["status"] == "ok", str(ergebnis_xgb.get("hinweis")))

print("\n=== Durchlauf: Wiederaufnahme ===")
check("vorher nichts fertig", rm.read_done() == set(), str(rm.read_done()))
rm.append_row(ergebnis)
rm.append_row(ergebnis_xgb)
fertig = rm.read_done()
check("beide Kombinationen erkannt",
      fertig == {("mini", "rf", "n10000"), ("mini", "xgb", "n10000")}, str(sorted(fertig)))
check("andere Variante gilt nicht als fertig", ("mini", "rf", "voll") not in fertig)
with open(rm.OUT_FILE, newline="", encoding="utf-8-sig") as handle:
    zeilen_erg = list(csv.DictReader(handle, delimiter=";"))
check("Kopfzeile nur einmal", len(zeilen_erg) == 2, str(len(zeilen_erg)))

# Ein Lauf ohne TABPFN_TOKEN schreibt fuer jeden Datensatz eine Fehlerzeile.
# Wuerden die als fertig gelten, liefe der naechste Lauf nach dem Eintragen
# des Tokens ins Leere und die Tabelle bliebe ohne TabPFN.
rm.append_row({
    "slug": "mini", "modell": "tabpfn", "variante": "n10000",
    "status": "fehler", "hinweis": "TabPFNLicenseError: kein Token",
})
fertig_nach_fehler = rm.read_done()
check("Fehlerzeile gilt nicht als fertig",
      ("mini", "tabpfn", "n10000") not in fertig_nach_fehler,
      str(sorted(fertig_nach_fehler)))
check("erfolgreiche Zeilen bleiben fertig",
      ("mini", "rf", "n10000") in fertig_nach_fehler)

print("\n=== Durchlauf: unpassende Metrik wird gemeldet ===")
schief = rm.rechne("mini", "rf", "n10000", zeile_mini, "RMSE")
check("Status metrik_passt_nicht", schief["status"] == "metrik_passt_nicht", schief["status"])
check("nichts gerechnet", schief["wert_mittel"] == "")

print("\n=== Durchlauf: Umgebung wird festgehalten ===")
check("umgebung ist Ergebnisspalte", "umgebung" in rm.COLUMNS)
u = rm.umgebung()
check("Umgebung nennt beide Versionen",
      "skl" in u and "xgb" in u and "py" in u, u)
check("Umgebung steht in der Ergebniszeile", ergebnis["umgebung"] == u, ergebnis["umgebung"])

print("\n=== Durchlauf: Ausgabepfad ===")
argv_vorher = sys.argv
ausgabe_dir = WORK / "drive_ordner"
ausgabe_dir.mkdir(exist_ok=True)
try:
    # Ein Ordner als --ausgabe muss zur Datei darin werden, sonst versucht das
    # Skript, auf den Ordner selbst zu schreiben.
    sys.argv = ["run_models.py", "--only", "gibtesnicht", "--ausgabe", str(ausgabe_dir)]
    try:
        rm.main()
    except SystemExit:
        pass
    check("Ordner wird zur Datei darin",
          rm.OUT_FILE == ausgabe_dir / "model_results.csv", str(rm.OUT_FILE))

    datei = WORK / "woanders.csv"
    sys.argv = ["run_models.py", "--only", "gibtesnicht", "--ausgabe", str(datei)]
    try:
        rm.main()
    except SystemExit:
        pass
    check("Dateipfad bleibt wie angegeben", rm.OUT_FILE == datei, str(rm.OUT_FILE))

    sys.argv = ["run_models.py", "--only", "gibtesnicht", "--daten", str(WORK / "anders")]
    try:
        rm.main()
    except SystemExit:
        pass
    check("Datenordner wird uebernommen", rm.DATA_DIR == WORK / "anders", str(rm.DATA_DIR))
finally:
    sys.argv = argv_vorher

print("\n=== Durchlauf: Download bei Bedarf ===")
rm.DATA_DIR = WORK / "mdata"
echt_download, echt_token = dl.download, dl.read_token
try:
    dl.read_token = lambda: {"Authorization": "Bearer egal"}
    dl.download = lambda slug, headers: ("regeln_fehlen", 0.0, 0.0)
    try:
        rm.hole_daten("irgendwas")
        check("fehlgeschlagener Download wirft Fehler", False, "kein Fehler")
    except RuntimeError as exc:
        check("fehlgeschlagener Download wirft Fehler", "regeln_fehlen" in str(exc), str(exc)[:60])

    dl.download = lambda slug, headers: ("ok", 1.5, 2.5)
    check("geglueckter Download gibt Zeit und Groesse", rm.hole_daten("x") == (1.5, 2.5))
    dl.download = lambda slug, headers: ("uebersprungen", 0.0, 0.0)
    check("vorhandene Daten gelten als Erfolg", rm.hole_daten("x") == (0.0, 0.0))
finally:
    dl.download, dl.read_token = echt_download, echt_token

print("\n=== Raenge: Richtung wird beachtet ===")
import aggregate_ranks as ar  # noqa: E402

werte_t = pd_.DataFrame(
    {"a": [0.9, 10.0], "b": [0.8, 20.0], "c": [0.7, 30.0]},
    index=["gross_gut", "klein_gut"],
)
richtung_t = pd_.Series({"gross_gut": 1, "klein_gut": -1})
r = ar.raenge(werte_t, richtung_t)
# Bei richtung +1 ist der groesste Wert Rang 1, bei -1 der kleinste. Genau
# hier waere ein Vorzeichenfehler in der Auswertung unsichtbar.
check("hoeher ist besser: groesster Wert bekommt Rang 1",
      list(r.loc["gross_gut"]) == [1.0, 2.0, 3.0], str(list(r.loc["gross_gut"])))
check("kleiner ist besser: kleinster Wert bekommt Rang 1",
      list(r.loc["klein_gut"]) == [1.0, 2.0, 3.0], str(list(r.loc["klein_gut"])))

gleich = pd_.DataFrame({"a": [0.5], "b": [0.5], "c": [0.1]}, index=["x"])
r_gleich = ar.raenge(gleich, pd_.Series({"x": 1}))
check("Bindungen bekommen den mittleren Rang",
      list(r_gleich.loc["x"]) == [1.5, 1.5, 3.0], str(list(r_gleich.loc["x"])))

print("\n=== Kritische Differenz und Gruppen ===")
# Nachgerechnet: 2.569 * sqrt(4*5 / (6*13)) = 1.3009...
cd_t = ar.kritische_differenz(4, 13)
check("CD fuer 4 Modelle und 13 Datensaetze", abs(cd_t - 1.3009) < 1e-3, "{:.4f}".format(cd_t))
check("kein CD fuer sehr viele Modelle", ar.kritische_differenz(42, 13) is None)

mittel_t = pd_.Series({"a": 1.0, "b": 1.5, "c": 3.0})
check("nur die nahen Modelle bilden eine Gruppe",
      ar.gruppen(mittel_t, 1.0) == [["a", "b"]], str(ar.gruppen(mittel_t, 1.0)))
check("bei grossem CD sind alle in einer Gruppe",
      ar.gruppen(mittel_t, 5.0) == [["a", "b", "c"]], str(ar.gruppen(mittel_t, 5.0)))
check("bei winzigem CD keine Gruppe", ar.gruppen(mittel_t, 0.01) == [])

print("\n=== Raenge: unvollstaendige Datensaetze ===")
rang_datei = WORK / "ergebnisse_rang.csv"
kopf = ("slug;modell;variante;status;wert_mittel;richtung;metrik_code;umgebung")
zeilen_rang = [kopf]
for slug, wert in (("d1", 0.9), ("d2", 0.8), ("d3", 0.7)):
    for m in ("rf", "xgb", "tabpfn"):
        # Bei d3 fehlt tabpfn - dieser Datensatz darf nicht mitgezaehlt werden,
        # sonst verschieben sich die Raenge der anderen Modelle.
        if slug == "d3" and m == "tabpfn":
            continue
        zeilen_rang.append("{};{};n10000;ok;{};1;auc;test".format(slug, m, wert))
rang_datei.write_text("\n".join(zeilen_rang) + "\n", encoding="utf-8-sig")

werte_g, richtung_g, metrik_g, fehlend_g, umg_g = ar.lade_ergebnisse(
    rang_datei, "n10000", None)
check("unvollstaendiger Datensatz faellt raus", fehlend_g == ["d3"], str(fehlend_g))
check("zwei vollstaendige bleiben", len(werte_g) == 2, str(len(werte_g)))
check("Umgebung wird gemeldet", umg_g == ["test"], str(umg_g))

# Nach einem Lauf mit --neu steht dieselbe Kombination zweimal in der Datei.
# pivot_table wuerde beide mitteln: aus 0.9 und 0.5 wuerde 0.7 - ein Wert, den
# kein Lauf gemessen hat. Gelten soll die letzte Zeile.
doppelt_datei = WORK / "ergebnisse_doppelt.csv"
zeilen_doppelt = list(zeilen_rang) + ["d1;rf;n10000;ok;0.5;1;auc;test"]
doppelt_datei.write_text("\n".join(zeilen_doppelt) + "\n", encoding="utf-8-sig")
werte_d, _, _, _, _ = ar.lade_ergebnisse(doppelt_datei, "n10000", None)
check("Doppelzeile wird nicht gemittelt",
      abs(werte_d.loc["d1", "rf"] - 0.5) < 1e-9, str(werte_d.loc["d1", "rf"]))

fehlt_datei = WORK / "ohne_richtung.csv"
fehlt_datei.write_text("slug;modell;variante;status;wert_mittel;metrik_code\n", encoding="utf-8-sig")
try:
    ar.lade_ergebnisse(fehlt_datei, "n10000", None)
    check("fehlende Spalte wird gemeldet", False, "kein Fehler")
except SystemExit as exc:
    check("fehlende Spalte wird gemeldet", "richtung" in str(exc), str(exc)[:60])

print("\n=== Raenge: ganzer Lauf schreibt Tabelle und Bild ===")
rang_aus = WORK / "rang_ausgabe"
argv_vorher = sys.argv
try:
    sys.argv = ["aggregate_ranks.py", "--ergebnisse", str(rang_datei),
                "--ausgabe", str(rang_aus)]
    ar.main()
finally:
    sys.argv = argv_vorher
check("Rangtabelle geschrieben", (rang_aus / "ranks_n10000.csv").is_file())
check("Diagramm geschrieben", (rang_aus / "cd_n10000.png").is_file())
with open(rang_aus / "ranks_n10000.csv", newline="", encoding="utf-8-sig") as handle:
    rang_zeilen = list(csv.DictReader(handle, delimiter=";"))
check("eine Zeile je vollstaendigem Datensatz", len(rang_zeilen) == 2, str(len(rang_zeilen)))
check("Metrik steht in der Rangtabelle", rang_zeilen[0]["metrik_code"] == "auc")

# Im Testdatensatz liegen alle Modelle gleich. Friedman gibt dann nan zurueck,
# und ein naiver Vergleich wuerde daraus einen belegten Unterschied machen.
import io as io_  # noqa: E402
import contextlib  # noqa: E402

puffer = io_.StringIO()
argv_vorher = sys.argv
try:
    sys.argv = ["aggregate_ranks.py", "--ergebnisse", str(rang_datei),
                "--ausgabe", str(rang_aus)]
    with contextlib.redirect_stdout(puffer):
        ar.main()
finally:
    sys.argv = argv_vorher
ausgabe_text = puffer.getvalue()
check("nan wird nicht als Unterschied gelesen",
      "Kein Testergebnis" in ausgabe_text and "unterscheiden sich" not in ausgabe_text,
      str([z.strip() for z in ausgabe_text.splitlines() if "Friedman" in z][:1]))

server.shutdown()
shutil.rmtree(WORK, ignore_errors=True)

print("\n" + "=" * 55)
if failures:
    print("FEHLGESCHLAGEN: {} von {}: {}".format(len(failures), len(failures), failures))
    sys.exit(1)
print("Alle Pruefungen bestanden.")
