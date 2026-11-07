# Kaggle Playground – Datenbeschaffung

Fünf Skripte für Forschungsfrage 1:

1. **[`scripts/download_competitions.py`](scripts/download_competitions.py)** —
   lädt die Datensätze aller Playground-Wettbewerbe über die Kaggle-API und
   protokolliert die Downloadzeit pro Datensatz.
2. **[`scripts/extract_features.py`](scripts/extract_features.py)** —
   liest die geladenen Datensätze und schreibt Basis-Merkmale
   (Zeilen, Features, Datentypen, fehlende Werte, Zielspalte) in eine CSV.
3. **[`scripts/merge_basicdata.py`](scripts/merge_basicdata.py)** —
   führt Wettbewerbsübersicht, extrahierte Merkmale und Download-Log in einer
   Tabelle zusammen.
4. **[`scripts/run_models.py`](scripts/run_models.py)** —
   lässt Random Forest, XGBoost, TabPFN und TabFM ungetunt über die
   Datensätze laufen und bewertet mit der Metrik, die Kaggle für den
   jeweiligen Wettbewerb vorgibt.
5. **[`scripts/aggregate_ranks.py`](scripts/aggregate_ranks.py)** —
   fasst die Ergebnisse über alle Datensätze zu mittleren Rängen zusammen und
   prüft die Unterschiede statistisch.

## Setup

```bash
pip install -r requirements.txt
```

Token holen unter [kaggle.com/settings](https://www.kaggle.com/settings) →
Abschnitt **API** → **Create New Token**. Beginnt mit `KGAT_`. Gesucht wird
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

Für die Datensätze und für Wald und Boosting ist das alles. Nur TabPFN braucht
zusätzlich einen Prior-Labs-Key in `TABPFN_TOKEN`, und zwar allein zur
Lizenzprüfung — siehe [Zur TabPFN-Version](#die-fundamentmodelle).

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
| `n_zielspalten` | Anzahl der erkannten Zielspalten |
| `ziel_dtype`, `ziel_n_unique` | Datentyp und Anzahl verschiedener Werte |
| `aufgabe_vermutet` | `binaer`, `multiklasse`, `regression`, `multi_label` |
| `n_numerisch`, `n_kategorial` | Datentypen der Features |
| `n_time` | Anzahl der als Datum/Zeit erkannten Features |
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

Ein Lauf mit `--only` aktualisiert nur die betroffenen Zeilen; die übrigen
Datensätze bleiben in der Tabelle stehen.

**Zu `n_features`:** Herausgerechnet werden nur die Zielspalte(n) und eine
Spalte namens `id`. Weitere Identifikatoren zählen mit — bei S4E1 etwa
`CustomerId` und `Surname`. Die lassen sich nicht zuverlässig automatisch
erkennen. Über die Eindeutigkeit geht es nicht: Bei S4E1 ist `CustomerId`
nur zu 14 % und `Surname` nur zu 1,7 % eindeutig, weil die synthetischen
Daten Dubletten enthalten — während `EstimatedSalary` auf 33 % kommt. Welche
Spalten ins Modell gehören, ist damit eine Entscheidung fürs Preprocessing
(Gliederungspunkt 3.3), nicht für dieses Skript. `n_features` ist eine obere
Schranke.

### Basisdaten zusammenführen

```bash
python scripts/merge_basicdata.py
```

Schreibt `metadata/basicdata.csv` mit einer Zeile pro Wettbewerb aus
`playground_overview.csv`. Die Tabellen werden über den Slug verbunden
(`slugs` in der Übersicht, `slug` in den anderen Dateien); dadurch bleiben
auch Wettbewerbe ohne heruntergeladene Daten oder extrahierte Merkmale
enthalten. Da `download_log.csv` Einträge anhängt, wird daraus pro Wettbewerb
der letzte Logeintrag verwendet. Fehlende Werte bleiben leer.

## 3. Modelle durchlaufen lassen

```bash
# Hauptauswertung: alle Modelle auf 10.000 Zeilen
python scripts/run_models.py

# einzelner Datensatz
python scripts/run_models.py --only s5e3

# zweite Variante: volle Datenmenge, nur Wald und Boosting
python scripts/run_models.py --variante voll
```

Bewertet wird per 5-facher Kreuzvalidierung auf `train.csv`. Die Testlabels
sind bei Kaggle nicht öffentlich, eine Auswertung gegen `test.csv` ist also
nicht möglich. Jeder Datensatz wird mit **der Metrik bewertet, die Kaggle für
diesen Wettbewerb vorgibt** — aus der Spalte `Metrik` der Übersichtstabelle.

### Gerechnet wird auf Colab

Die ganze Auswertung läuft über
[`notebooks/colab_run_models.ipynb`](notebooks/colab_run_models.ipynb), nicht
lokal. Grund: Die Metrikwerte sind bei festem Startwert auf jedem Rechner
gleich, die **Zeiten** nicht — und TabPFN braucht ohnehin eine GPU. Käme ein
Teil der Tabelle von einem Laptop und ein Teil von Colab, wären die Laufzeiten
nicht vergleichbar und ein Versionsunterschied bei XGBoost würde unbemerkt
andere Zahlen liefern.

Deshalb hält jede Ergebniszeile in der Spalte `umgebung` fest, wo sie
entstanden ist (`colab py3.12.10 skl1.7.2 xgb3.0.5`). Gemischte Messungen
fallen damit auf, statt still in der Tabelle zu liegen.

Das Notebook holt jeden Datensatz von Kaggle, rechnet ihn und löscht ihn
wieder — auf der Platte liegt nie mehr als einer, auf Drive landet nur die
Ergebnis-CSV. Die 3,7 GB müssen also nirgendwo in die Cloud. Von Kaggle in ein
Google-Rechenzentrum zu laden ist außerdem schneller als durch eine private
Leitung.

Lokal laufen die Skripte weiterhin, das ist zum Entwickeln und Debuggen
gedacht. Solche Läufe gehören nicht in die Ergebnistabelle.

| Flag | Zweck |
|---|---|
| `--ausgabe` | Pfad der Ergebnis-CSV; ein Ordner wird zu `<ordner>/model_results.csv` |
| `--daten` | Ordner mit den Datensätzen |
| `--laden` | fehlende Datensätze von Kaggle holen |
| `--daten-loeschen` | Datensatz nach der Rechnung entfernen (nur mit `--laden`) |
| `--neu` | fertige Kombinationen neu rechnen |

### Zwei Varianten

Die Fundamentmodelle lesen ihre Trainingsdaten als Kontext, statt daraus zu
lernen, und sind darin begrenzt: TabFM liest voreingestellt nur 100 Zeilen.
Von den 34 geladenen Datensätzen liegt genau einer unter 10.000
(S5E3 mit 2.190 Zeilen). Daraus folgen zwei Durchläufe:

| Variante | Zeilen | Modelle | Zweck |
|---|---|---|---|
| `n10000` | auf 10.000 verkleinert | alle | **Hauptauswertung.** Alle Modelle sehen dieselbe Datenmenge, der Vergleich ist fair. |
| `voll` | alle | Wald, Boosting | Zeigt, was die Deckelung auf 10.000 Zeilen kostet. |

Verkleinert wird bei Klassifikation geschichtet, damit seltene Klassen
erhalten bleiben, und mit festem Startwert — alle Modelle sehen damit genau
dieselben Zeilen. Bei 10.000 Zeilen und fünf Folds trainiert ein Modell auf
8.000 Zeilen.

Der Grund für die Deckelung ist **nicht** eine technische Grenze, sondern die
Vergleichbarkeit: Alle Modelle sollen dieselbe Datenmenge sehen. TabPFN-3.5
könnte rein technisch bis zu einer Million Zeilen verarbeiten, TabFM liest
voreingestellt 100. Ohne Deckelung würde man also messen, wie viel Kontext ein
Modell zulässt, und nicht, wie gut es damit umgeht. Die 10.000 sind damit eine
methodische Entscheidung, die in der Arbeit so begründet werden sollte.

### Die Fundamentmodelle

| `--modelle` | Modell | Gewichte | Grenzen |
|---|---|---|---|
| `rf` | Random Forest (scikit-learn) | — | — |
| `xgb` | XGBoost | — | — |
| `tabpfn` | [TabPFN](https://github.com/PriorLabs/TabPFN) | TabPFN-3.5 | GPU nötig (auf CPU Grenze bei 5.000 Zeilen) |
| `tabfm` | [TabFM](https://github.com/google-research/tabfm) (Google Research) | v1.0.0 | 100 Kontextzeilen, 500 Merkmale |

Beide Fundamentmodelle sind scikit-learn-kompatibel und werden erst beim
Aufruf importiert — ein Lauf mit nur `rf` und `xgb` braucht weder PyTorch noch
Modellgewichte.

**Zur TabPFN-Version:** Verwendet wird **TabPFN-3.5**, der aktuelle Stand. Die
Gewichtsdatei heißt `tabpfn-v3.5-20260909.safetensors` und trägt damit ein Datum
im Namen — der gemessene Stand ist dadurch zitierbar. Er steht zusätzlich in
jeder Ergebniszeile in der Spalte `modell_version`; bei Fundamentmodellen
entscheidet er über das Ergebnis.

Prior Labs verlangt für die Gewichte ab TabPFN-2.5 einmalig, dass die Lizenz
(bei 3.5: `tabpfn-3-5-license-v1.0`) im Konto auf <https://ux.priorlabs.ai>
bestätigt wird. Geprüft wird das über einen API-Key, der in der
Umgebungsvariablen `TABPFN_TOKEN` erwartet wird; in Colab kommt er aus einem
Secret gleichen Namens. Die Gewichtsdatei selbst liegt offen auf Hugging Face,
ein Konto dort ist nicht nötig.

**Gerechnet wird in jedem Fall lokal.** Der Key dient allein der Lizenzprüfung,
es gehen keine Daten an Prior Labs. Das wäre das separate Paket `tabpfn-client`,
das die Berechnung auf deren Servern ausführt und das wir nicht verwenden — es
bedient laut Prior Labs andere Modelllinien (`Plus`, `Thinking`) als die
veröffentlichten Gewichte und wäre für eine Arbeit nicht nachprüfbar.

Wer keinen Key hat, kommt mit `--tabpfn-version V2` ohne jede Anmeldung
durch. TabPFN-2 steht unter der Prior Labs License (Apache 2.0 mit
Namensnennung); auf CPU gilt dort allerdings schon ab 1.000 Zeilen eine Grenze.

**Getrennte Umgebungen nötig:** TabFM legt seine Abhängigkeiten fest auf
`scikit-learn==1.6.0` und `torch==2.12.1`. Unsere Zahlen für Wald und XGBoost
entstehen unter scikit-learn 1.7.2, und genau dessen Voreinstellungen sind der
Untersuchungsgegenstand. TabFM darf deshalb **nicht** in derselben Laufzeit
installiert werden wie ein `rf`- oder `xgb`-Lauf. Das Notebook trennt das; die
Ergebnisdatei wird nur angehängt, es geht dabei nichts verloren.

**Was „ungetunt" bei einem Fundamentmodell heißt, ist nicht dasselbe wie bei
einem Wald.** TabFM liest voreingestellt nur **100 Kontextzeilen** und bildet
über mehrere gezogene Kontexte ein Ensemble. Es sieht damit von Haus aus
deutlich weniger Daten als Wald und XGBoost auf denselben 8.000 Zeilen. Ob man
das so stehen lässt — es ist die Voreinstellung, also „ungetunt" im selben
Sinne — oder `max_num_rows` anhebt, ist eine Entscheidung für die Methodik und
sollte in der Arbeit begründet werden, nicht stillschweigend fallen.

### Was „ungetunt“ heißt

Gesetzt werden nur `random_state` und `n_jobs`. Beides sind keine
Hyperparameter — das eine macht den Lauf wiederholbar, das andere nutzt die
vorhandenen Kerne. Alles andere bleibt auf der Voreinstellung der Bibliothek.

Deshalb stehen `scikit-learn` und `xgboost` in `requirements.txt` mit `==`
statt `>=`: Die Voreinstellungen *sind* hier der Untersuchungsgegenstand.

### Vorverarbeitung

Für alle Modelle identisch und je Fold nur auf den Trainingsdaten angepasst:

- Spalte `id` fliegt raus, Zeilen ohne Zielwert ebenfalls
- Zahlen: fehlende Werte durch den Median ersetzt
- Text: Ordinalkodierung, fehlende Werte sind die eigene Kategorie `nan`
- unbekannte Kategorien im Testfold werden zu `-1`

Das ist bewusst minimal. Vergleicht man Modelle, darf die Vorverarbeitung
nicht mitvariieren, sonst vergleicht man Vorverarbeitungen.

**Eine Einschränkung dadurch:** XGBoost kann fehlende Werte selbst
verarbeiten, der Wald nicht. Weil beide dieselben Daten bekommen müssen, wird
auch für XGBoost imputiert — es tritt hier also ohne einen seiner Vorteile an.
Betroffen sind 11 der 34 Datensätze.

### Metriken

Der Freitext der Übersichtstabelle wird auf einen Code abgebildet
([`scripts/metrics.py`](scripts/metrics.py)). Ein unbekannter Text führt zu
einem Fehler, nicht zu einer stillen Ersatzmetrik — sonst fällt eine geänderte
Tabelle nicht auf.

| Metrik | Code | besser ist | Datensätze |
|---|---|---|---:|
| AUC | `auc` | größer | 13 |
| RMSE | `rmse` | kleiner | 6 |
| Accuracy | `accuracy` | größer | 4 |
| RMSLE | `rmsle` | kleiner | 3 |
| Balanced Accuracy | `balanced_accuracy` | größer | 3 |
| MAPE | `mape` | kleiner | 1 |
| R² | `r2` | größer | 1 |
| MCC | `mcc` | größer | 1 |
| MAP@3 | `map3` | größer | 1 |
| AUC über 7 Targets | `auc_multilabel` | größer | 1 |

Die Spalte `richtung` (`+1`/`-1`) steht in jeder Ergebniszeile. Ohne sie lässt
sich über Datensätze mit verschiedenen Metriken hinweg nicht ranken.

Zwei Sonderfälle: **MAP@3** ist selbst gerechnet, weil scikit-learn das nicht
anbietet — bei genau einer richtigen Klasse pro Zeile ist das `1/Rang`, wenn
die Klasse unter den ersten drei liegt, sonst 0. **RMSLE** verträgt keine
negativen Vorhersagen, die werden bei 0 abgeschnitten; ein ungetuntes
Regressionsmodell liefert durchaus negative Werte.

Bei **S4E3** (7 Zielspalten) wird je Zielspalte ein eigenes Modell trainiert
und über die 7 AUC-Werte gemittelt — genau so bewertet Kaggle dort.

### Ergebnis

```
results/model_results.csv
```

Eine Zeile pro Kombination aus Datensatz, Modell und Variante. Wichtige
Spalten: `metrik_code`, `richtung`, `wert_mittel`, `wert_std`, `werte_folds`
(die fünf Einzelwerte, `|`-getrennt), `n_zeilen_genutzt`, `sekunden_fit`,
`status`, `hinweis` und `umgebung`.

Jede fertige Kombination wird **sofort angehängt** und beim nächsten Lauf
übersprungen. Eine abgebrochene Colab-Sitzung kostet also nur den gerade
laufenden Datensatz. Mit `--neu` wird neu gerechnet.

Übersprungen wird nur, was `status = ok` hat. Eine Fehlerzeile ist kein
Ergebnis und wird beim nächsten Lauf erneut versucht — sonst würde ein Lauf,
der etwa am fehlenden `TABPFN_TOKEN` scheitert, alle Kombinationen als „fertig"
markieren und die Tabelle bliebe stillschweigend ohne TabPFN. Die Fehlerzeilen
bleiben in der Datei stehen; die Auswertung filtert sie. Steht dieselbe
Kombination nach einem `--neu`-Lauf zweimal drin, gilt die letzte Zeile.

`status` ist `ok`, `fehler`, `metrik_passt_nicht` (Metrik und erkannter
Aufgabentyp gehen nicht zusammen) oder `metrik_unmoeglich` (z. B. RMSLE bei
negativen Zielwerten).

## 4. Ergebnisse vergleichen

```bash
python scripts/aggregate_ranks.py
python scripts/aggregate_ranks.py --variante voll
python scripts/aggregate_ranks.py --modelle rf xgb tabpfn
```

Hier wird aus 34 Datensätzen mit zehn verschiedenen Metriken **eine** Aussage.
Mitteln geht nicht: Ein AUC von 0,9 und ein RMSE von 87.534 haben keinen
gemeinsamen Nenner. Vergleichbar sind die **Ränge** — pro Datensatz bekommt das
beste Modell Rang 1, darüber wird der Mittelwert gebildet. Dafür ist die Spalte
`richtung` da: Bei AUC ist groß gut, bei RMSE klein.

Danach die Tests, die für genau diesen Fall vorgesehen sind (Demšar 2006,
*Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR):

| Test | wofür |
|---|---|
| **Friedman** | Unterscheiden sich die Modelle überhaupt? (ab 3 Modellen) |
| **Wilcoxon** | dasselbe für genau 2 Modelle |
| **Nemenyi** | Welche Paare unterscheiden sich? Über die kritische Differenz CD |

Ein Datensatz, bei dem ein Modell fehlt, wird ausgelassen und beim Namen
genannt — sonst verschieben sich die Ränge der übrigen Modelle. Stammen die
Ergebnisse aus mehreren `umgebung`-Werten, warnt das Skript.

### Ergebnis

```
results/ranks_<variante>.csv
results/cd_<variante>.png
```

Im CD-Diagramm liegen Modelle unter einem gemeinsamen Balken enger beieinander
als die kritische Differenz — ihr Unterschied ist dann **nicht belegt**. Das ist
der Punkt, an dem die Auswertung ehrlich wird: Ein Vorsprung in der Rangliste
allein ist noch keine Aussage. Bei 13 Datensätzen und zwei Modellen gewinnt der
Wald neunmal, und der Wilcoxon-Test gibt dafür trotzdem nur p = 0,27.

## Welche Wettbewerbe?

Aktuell 83 Wettbewerbe: 23 Tabular Playground Series (2021–2022),
26 in Season 3, je 12 in Season 4 und 5, 10 in Season 6.
