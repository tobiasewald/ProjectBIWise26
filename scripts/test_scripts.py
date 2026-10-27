#!/usr/bin/env python3
"""Testet die beiden Skripte gegen einen lokalen HTTP-Server.

Kaggle wird nicht kontaktiert, ein Token ist nicht noetig.

    python scripts/test_scripts.py
"""

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

server.shutdown()
shutil.rmtree(WORK, ignore_errors=True)

print("\n" + "=" * 55)
if failures:
    print("FEHLGESCHLAGEN: {} von {}: {}".format(len(failures), len(failures), failures))
    sys.exit(1)
print("Alle Pruefungen bestanden.")
