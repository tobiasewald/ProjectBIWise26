#!/usr/bin/env python3
"""Laedt die Datensaetze aller Kaggle-Playground-Wettbewerbe herunter.

Die Wettbewerbe kommen aus der Spalte 'slugs' in
metadata/playground_overview.csv. Pro Wettbewerb wird das Archiv geladen,
nach data/<slug>/ entpackt und eine Zeile an logs/download_log.csv angehaengt.

    python scripts/download_competitions.py --rules-urls   # Links zum Akzeptieren
    python scripts/download_competitions.py --limit 1      # erstmal einen testen
    python scripts/download_competitions.py                # alle
"""

import argparse
import csv
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OVERVIEW = ROOT / "metadata" / "playground_overview.csv"
DATA_DIR = ROOT / "data"
LOG_FILE = ROOT / "logs" / "download_log.csv"
API = "https://www.kaggle.com/api/v1/competitions/data/download-all/"
PLACEHOLDER = "KGAT_HIER_DEIN_TOKEN_EINFUEGEN"

# Reihenfolge der Tokenquellen. Path.home() loest sich bei jedem Nutzer auf
# seinen eigenen Ordner auf, da steht nichts fest verdrahtet.
PROJECT_TOKEN_FILE = ROOT / "access_token"
TOKEN_FILE = Path.home() / ".kaggle" / "access_token"


def read_token():
    """Liest das Kaggle-Token und gibt den Authorization-Header zurueck."""
    token = (os.environ.get("KAGGLE_API_TOKEN") or "").strip()
    source = "KAGGLE_API_TOKEN"

    for path in (PROJECT_TOKEN_FILE, TOKEN_FILE):
        if token:
            break
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            source = str(path)

    if token == PLACEHOLDER:
        sys.exit(
            "In " + source + " steht noch der Platzhalter.\n"
            "Bitte durch das echte Token ersetzen "
            "(kaggle.com/settings -> API -> 'Create New Token')."
        )
    if not token.startswith("KGAT_") or len(token) < 25:
        sys.exit(
            "Kein gueltiges Kaggle-Token gefunden. Es wird gesucht in:\n"
            "  1. Umgebungsvariable KAGGLE_API_TOKEN\n"
            "  2. " + str(PROJECT_TOKEN_FILE) + "\n"
            "  3. " + str(TOKEN_FILE) + "\n"
            "Token holen: kaggle.com/settings -> API -> 'Create New Token'\n"
            "(nur das Token hineinschreiben, es beginnt mit 'KGAT_')"
        )
    print("Token aus: " + source)
    return {"Authorization": "Bearer " + token}


def read_slugs():
    """Liest die Wettbewerbs-Slugs aus der Uebersichtstabelle."""
    text = OVERVIEW.read_text(encoding="utf-8")
    rows = csv.DictReader(text.splitlines(), delimiter=";")
    return [
        row["slugs"].strip()
        for row in rows
        if re.fullmatch(r"[a-z0-9-]+", (row.get("slugs") or "").strip())
    ]


def download(slug, headers):
    """Laedt einen Wettbewerb. Gibt (status, sekunden, megabytes) zurueck."""
    target = DATA_DIR / slug
    if (target / ".done").exists():
        return "uebersprungen", 0.0, 0.0

    start = time.perf_counter()
    response = requests.get(API + slug, headers=headers, stream=True, timeout=120)
    if response.status_code == 403:
        # Die Wettbewerbsregeln muessen einmalig im Browser akzeptiert werden.
        return "regeln_fehlen", 0.0, 0.0
    if response.status_code != 200:
        return "http_" + str(response.status_code), 0.0, 0.0

    # Erst anlegen, wenn wirklich Daten kommen - sonst bleiben leere Ordner liegen.
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "archive.zip"
    with open(archive, "wb") as sink:
        for chunk in response.iter_content(1024 * 1024):
            sink.write(chunk)
    seconds = time.perf_counter() - start
    megabytes = archive.stat().st_size / 1_048_576

    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            # Kein Entpacken ausserhalb des Zielordners.
            if name.startswith("/") or ".." in name.replace("\\", "/").split("/"):
                raise ValueError("Unsicherer Pfad im Archiv: " + name)
        zf.extractall(target)

    archive.unlink()
    (target / ".done").write_text("ok\n", encoding="utf-8")
    return "ok", seconds, megabytes


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", help="nur Slugs mit diesem Teilstring")
    parser.add_argument("--limit", type=int, help="nur die ersten N Wettbewerbe")
    parser.add_argument(
        "--rules-urls",
        action="store_true",
        help="nur die Links zum Akzeptieren der Wettbewerbsregeln ausgeben",
    )
    args = parser.parse_args()

    slugs = read_slugs()
    if args.only:
        needles = [n.lower() for n in args.only]
        slugs = [s for s in slugs if any(n in s for n in needles)]
    if args.limit:
        slugs = slugs[: args.limit]
    if not slugs:
        sys.exit("Keine Wettbewerbe ausgewaehlt.")

    if args.rules_urls:
        for slug in slugs:
            print("https://www.kaggle.com/competitions/" + slug + "/rules")
        return

    headers = read_token()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Anhaengen, damit Messwerte aus frueheren Laeufen erhalten bleiben.
    new_file = not LOG_FILE.exists()
    counts = {}
    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as handle:
        log = csv.writer(handle, delimiter=";")
        if new_file:
            log.writerow(["slug", "status", "sekunden", "megabytes", "zeitpunkt"])

        for i, slug in enumerate(slugs, 1):
            print("[{}/{}] {} ... ".format(i, len(slugs), slug), end="", flush=True)
            try:
                status, seconds, megabytes = download(slug, headers)
            except Exception as exc:
                status, seconds, megabytes = "fehler", 0.0, 0.0
                print(type(exc).__name__ + ": " + str(exc)[:120])
            else:
                print("{} ({:.1f} s, {:.1f} MB)".format(status, seconds, megabytes))

            counts[status] = counts.get(status, 0) + 1
            log.writerow([
                slug,
                status,
                round(seconds, 1),
                round(megabytes, 1),
                datetime.now().isoformat(timespec="seconds"),
            ])
            handle.flush()

            if status == "http_401":
                sys.exit(
                    "\nKaggle lehnt das Token ab. Bitte ein neues erzeugen:\n"
                    "kaggle.com/settings -> API -> 'Create New Token'"
                )

    print("\n" + "-" * 50)
    for status in sorted(counts):
        print("  {:<16} {:>3}".format(status, counts[status]))
    print("  Log: " + str(LOG_FILE))

    if counts.get("regeln_fehlen"):
        print(
            "\n  Bei {} Wettbewerben fehlt die Regelzustimmung. Links mit:\n"
            "  python scripts/download_competitions.py --rules-urls".format(
                counts["regeln_fehlen"]
            )
        )


if __name__ == "__main__":
    main()
