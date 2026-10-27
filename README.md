# Kaggle Playground – Datenbeschaffung

Zwei Skripte für Forschungsfrage 1:

1. **[`scripts/download_competitions.py`](scripts/download_competitions.py)** —
   lädt die Datensätze aller Playground-Wettbewerbe über die Kaggle-API und
   protokolliert die Downloadzeit pro Datensatz.
2. **[`scripts/extract_features.py`](scripts/extract_features.py)** —
   liest die geladenen Datensätze und schreibt Basis-Merkmale
   (Zeilen, Features, Datentypen, fehlende Werte, Zielspalte) in eine CSV.

## Setup

```bash
pip install -r requirements.txt
```

Token holen unter [kaggle.com/settings](https://www.kaggle.com/settings) →
Abschnitt **API** → **Create New Token**. Es beginnt mit `KGAT_`. Gesucht wird
in dieser Reihenfolge:

| Priorität | Ort |
|---|---|
| 1 | Umgebungsvariable `KAGGLE_API_TOKEN` |
| 2 | `access_token` im Projektordner |
| 3 | `~/.kaggle/access_token` |

In die Datei kommt nur das Token, nichts weiter. Beide Dateinamen stehen in
`.gitignore`.

`~/.kaggle/access_token` ist Kaggles eigene Konvention — dort finden es auch
das offizielle `kaggle`-CLI und `kagglehub`, und das Token überlebt, wenn der
Projektordner neu geklont wird. Der Projektordner ist die Alternative, wenn
man alles an einer Stelle haben will. Der Pfad ist nicht fest verdrahtet,
sondern wird über `Path.home()` pro Nutzer aufgelöst.

Beim Start gibt das Skript nur aus, aus welcher *Quelle* das Token kam, nie den
Wert. Solange der Platzhalter drinsteht, sagt es das ausdrücklich, statt in
einen 401 zu laufen.

## 1. Daten herunterladen

```bash
# Links, um die Wettbewerbsregeln zu akzeptieren (siehe unten)
python scripts/download_competitions.py --rules-urls > rules_todo.txt

# erstmal einen Wettbewerb testen
python scripts/download_competitions.py --limit 1

# alle 83
python scripts/download_competitions.py

# nur bestimmte
python scripts/download_competitions.py --only s4e1 s5e12
```

Fertige Datensätze werden beim nächsten Lauf übersprungen (Marker `.done` im
Ordner). Nach einem Abbruch einfach denselben Befehl erneut starten.

**Wichtig:** Die Kaggle-API gibt **HTTP 403** zurück, solange die Regeln eines
Wettbewerbs nicht akzeptiert sind — auch bei längst beendeten Wettbewerben.
Das muss einmal pro Wettbewerb im Browser passieren ("Join Competition" /
"I Understand and Accept") und geht nicht über die API. Das Skript bricht
daran nicht ab, sondern vermerkt `regeln_fehlen` und macht weiter. Heißt
praktisch: einmal 83 Mal klicken.

### Ergebnis

```
data/<slug>/train.csv, test.csv, sample_submission.csv, .done
logs/download_log.csv
```

`logs/download_log.csv` ist semikolonsepariert und mit BOM, öffnet sich also
in deutschem Excel direkt richtig. Über `slug` lässt sich die Datei mit der
Übersichtstabelle verknüpfen.

| Spalte | Bedeutung |
|---|---|
| `slug` | Wettbewerbskennung, Schlüssel für den Join |
| `status` | `ok`, `uebersprungen`, `regeln_fehlen`, `http_<code>`, `fehler` |
| `sekunden` | **Downloadzeit** |
| `megabytes` | Größe des Archivs |
| `zeitpunkt` | wann der Versuch lief |

Das Log wird **angehängt**, nicht überschrieben — ein zweiter Lauf nach dem
Akzeptieren der Regeln löscht also die bereits gemessenen Zeiten nicht.

## 2. Merkmale extrahieren

```bash
python scripts/extract_features.py
```

Schreibt `metadata/dataset_features.csv`, eine Zeile pro Datensatz:

| Spalte | Bedeutung |
|---|---|
| `slug` | Schlüssel für den Join mit der Übersichtstabelle |
| `n_zeilen`, `n_spalten` | Größe von `train.csv` |
| `n_features` | Spalten ohne `id` und ohne Zielspalte |
| `zielspalte` | aus der Differenz train/test bestimmt (s. u.) |
| `ziel_dtype`, `ziel_n_unique` | Datentyp und Anzahl verschiedener Werte |
| `aufgabe_vermutet` | `binaer`, `multiklasse`, `regression`, `multi_label` |
| `n_numerisch`, `n_kategorial` | Datentypen der Features |
| `n_fast_eindeutig` | Features mit über 99 % verschiedenen Werten |
| `n_fehlende_werte` | fehlende Zellen in `train.csv` |
| `anteil_fehlend_prozent` | Anteil fehlender Zellen |
| `spalten_mit_fehlenden` | wie viele Spalten betroffen sind |
| `train_mb` | Dateigröße |

Die **Zielspalte** wird als die Spalte bestimmt, die in `train.csv` steht und
in `test.csv` fehlt. Bei den Playground-Wettbewerben ist das zuverlässig und
spart es, 83 Mal in der Wettbewerbsbeschreibung nachzusehen. Bei den
Multi-Label-Episoden (z. B. S4E3 mit 7 Defekttypen) sind es mehrere, dann
stehen sie `|`-getrennt in der Spalte.

`aufgabe_vermutet` ist eine Heuristik: zwei verschiedene Werte → binär,
numerisch mit mehr als 20 Werten → Regression, sonst Multiklasse. Für die
Methodik der Arbeit sollte das gegen die offizielle Metrik aus der
Übersichtstabelle geprüft werden.

**Zu `n_features`:** Herausgerechnet werden nur die Zielspalte(n) und eine
Spalte namens `id`. Weitere Identifikatoren zählen mit — bei S4E1 etwa
`CustomerId` und `Surname`. Die lassen sich nicht zuverlässig automatisch
erkennen: `n_fast_eindeutig` findet sie dort nicht, weil die synthetischen
Daten Dubletten enthalten (`CustomerId` ist nur zu 14 % eindeutig). Welche
Spalten ins Modell gehören, ist damit eine Entscheidung fürs Preprocessing
(Gliederungspunkt 3.3), nicht für dieses Skript. `n_features` ist eine obere
Schranke.

## Welche Wettbewerbe?

Die Liste kommt aus der Spalte `slugs` in
[`metadata/playground_overview.csv`](metadata/playground_overview.csv) — der
UTF-8-Konvertierung der gemeinsamen Excel-Übersicht. Trägt jemand dort eine
Zeile nach, zieht das Skript sie automatisch mit. Zeilen ohne Slug
(Dez 2022, Nov/Dez 2026 — dort gibt es keinen Wettbewerb) werden übersprungen.

Aktuell 83 Wettbewerbe: 23 Tabular Playground Series (2021–2022),
26 in Season 3, je 12 in Season 4 und 5, 10 in Season 6.

## Tests

```bash
python scripts/test_scripts.py
```

30 Prüfungen gegen einen lokalen HTTP-Server: Download, Entpacken,
Wiederaufnahme, die HTTP-Fälle 401/403/404, Abwehr von Pfaden, die aus dem
Zielordner herausführen, das Lesen des Tokens und die Merkmalsextraktion.
Kaggle wird dabei nicht kontaktiert, ein Token ist nicht nötig.

Gegen Kaggle selbst ist beides gelaufen: Season 4–6, 34 Wettbewerbe,
926,9 MB Archive in 137,1 s, entpackt 3,7 GB. Die Merkmalsextraktion lief
über alle 34 durch, und `aufgabe_vermutet` stimmt bei allen 34 mit der
offiziellen Metrik aus der Übersichtstabelle überein — die Heuristik ist
damit geprüft, nicht nur plausibel.

Season 3 und die Tabular Playground Series 2021–2022 (49 Wettbewerbe) sind
noch nicht geladen, dort fehlt die Regelzustimmung.

## Was bewusst fehlt

Keine Wiederholversuche bei Netzwerkfehlern, keine parallelen Downloads, keine
Größenbegrenzung. Ein fehlgeschlagener Wettbewerb steht als `fehler` im Log;
weil fertige Downloads übersprungen werden, reicht ein erneuter Lauf. Parallele
Downloads wären schneller, würden aber die gemessenen Zeiten verfälschen, weil
sich die Verbindungen die Bandbreite teilen.
