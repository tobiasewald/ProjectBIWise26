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
entstanden ist (`colab py3.13.18 skl1.6.1 xgb3.4.1`). Gemischte Messungen
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
lernen. Wie viel Kontext ein Modell verkraftet, unterscheidet sich dabei
erheblich — und genau deshalb wird gedeckelt.
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
Vergleichbarkeit: Alle Modelle sollen dieselbe Datenmenge sehen. Ohne
Deckelung bekäme jedes Modell eine andere Menge Kontext, und gemessen würde,
wie viel ein Modell verkraftet, statt wie gut es damit umgeht. Die 10.000 sind
damit eine methodische Entscheidung, die in der Arbeit so begründet werden
sollte.

Die Zahl stammt ursprünglich von TabPFN-2, wo 10.000 Zeilen die fest
verdrahtete Obergrenze waren (`MAX_NUMBER_OF_SAMPLES=10_000` in
`inference_config.py`; darüber wirft das Paket einen
`TabPFNValidationError`). Ab TabPFN-2.5 steht dort 50.000, bei 3.5 liegt der
Wert im Checkpoint selbst. Technisch bindet die Grenze also nicht mehr — als
gemeinsame Datenmenge für alle vier Modelle bleibt sie trotzdem eine
Entscheidung, die getroffen werden muss.

### Die Fundamentmodelle

| `--modelle` | Modell | Gewichte | Grenzen |
|---|---|---|---|
| `rf` | Random Forest (scikit-learn) | — | — |
| `xgb` | XGBoost | — | — |
| `tabpfn` | [TabPFN](https://github.com/PriorLabs/TabPFN) | TabPFN-3.5 | GPU nötig (auf CPU Grenze bei 5.000 Zeilen) |
| `tabfm` | [TabFM](https://github.com/google-research/tabfm) (Google Research) | v1.0.0 | 500 Merkmale; GPU nötig, und das Gerät muss gesetzt werden |
| `tabdpt` | [TabDPT](https://github.com/layer6ai-labs/TabDPT) (Layer 6 AI) | v1.3 | auf **echten** Tabellen vortrainiert; Schalter sitzen auf `predict()` |

Alle drei Fundamentmodelle werden erst beim Aufruf importiert — ein Lauf mit
nur `rf` und `xgb` braucht weder PyTorch noch Modellgewichte. TabPFN und TabFM
sind unmittelbar scikit-learn-kompatibel, TabDPT bekommt eine dünne Hülle,
siehe unten.

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

**Getrennte Umgebungen nötig:** TabFM bringt eigene Festlegungen für
`scikit-learn` und `torch` mit und verschiebt damit die Versionen, unter denen
Wald und XGBoost gemessen werden — und genau deren Voreinstellungen sind der
Untersuchungsgegenstand. TabFM darf deshalb **nicht** in derselben Laufzeit
installiert werden wie ein `rf`- oder `xgb`-Lauf. Das Notebook trennt das; die
Ergebnisdatei wird nur angehängt, es geht dabei nichts verloren.

Beobachtet am 15.11.2026 (Colab, torch 2.14.1+cu130 auf einer L4): Die
TabFM-Installation lässt `scikit-learn` auf 1.6.1, also auf genau dem Stand
aus `requirements.txt`. Der einzige Konflikt, den `pip` meldete, war
`inflect 7.6.0 requires typeguard>=4.0.1, but you have typeguard 2.13.3` —
TabFM setzt `typeguard` für `jaxtyping` fest, und `inflect` gehört zur
Colab-Grundausstattung und kommt in diesem Projekt nirgends vor. Der Konflikt
trifft uns also nicht.

Die Laufzeiten bleiben trotzdem getrennt. Das ist eine Beobachtung an einem Tag
mit einem Colab-Abbild, keine Zusage — und die Zelle im Notebook prüft die
Version nach der Installation, damit eine Verschiebung nicht unbemerkt bleibt.

**TabFM wählt sein Rechengerät nicht selbst.** `tabfm_v1_0_0.load()` hat
`device=None` voreingestellt, und das bedeutet laut Quelltext CPU — anders als
TabPFN, das eine vorhandene CUDA-Karte von sich aus nimmt. Zusammen mit
`dtype=torch.bfloat16`, das auf der CPU nur nachgebildet wird, rechnet ein
einzelner Datensatz dann minutenlang, während die zugeteilte GPU nichts tut.
`scripts/run_models.py` setzt das Gerät deshalb selbst und meldet beim Laden,
worauf gerechnet wird.

**Was „ungetunt" bei einem Fundamentmodell heißt, ist nicht dasselbe wie bei
einem Wald.** TabFM bildet über mehrere gezogene Kontexte ein Ensemble:
`n_estimators` steht voreingestellt auf **32**, `max_num_rows` auf **`None`**.
`None` heißt kein Zuschnitt — jedes der 32 Ensemble-Mitglieder liest die
vollen 8.000 Trainingszeilen als Kontext. Ungetuntes TabFM sieht also nicht
weniger Daten als Wald und XGBoost, sondern dieselben, nur 32-mal. Das macht
den Lauf um ein Vielfaches teurer als TabPFN. Dass `n_estimators` und
`max_num_rows` trotzdem auf der Voreinstellung bleiben, ist die konsequente
Lesart von „ungetunt"; ein Eingriff müsste in der Arbeit begründet werden.

> **Vorsicht bei TabFMs eigener README.** Deren FAQ nennt 100 Kontextzeilen
> als Voreinstellung. Im Quelltext von `TabFMClassifier` und `TabFMRegressor`
> steht `max_num_rows: Optional[int] = None`, und einen Ersatzwert gibt es an
> keiner Stelle der Datei. Gemessen wird, was der Code tut.

#### Was TabFM kostet, gemessen

Erster vollständiger Versuch am 14.11.2026 auf einer Colab-T4
(14,56 GiB, Rechenfähigkeit 7.5), `batch_size=8`. Von 34 Datensätzen ist
**einer** durchgelaufen:

| | S5E3 |
|---|---|
| Zeilen | 2.190 (kleinster Datensatz, unter der Deckelung) |
| AUC | 0,894161 — zwischen Wald (0,879) und TabPFN (0,897) |
| Zeit | **729 s** für fünf Folds |

Die übrigen 33 scheiterten an `CUDA out of memory`. Bei 10.000 Zeilen
verlangte TabFM eine einzelne Zuteilung von **23,84 GiB** — knapp 3 GiB je
Ensemble-Mitglied, auf einer Karte mit 14,56 GiB.

Beides skaliert quadratisch mit der Kontextgröße, was die zwei Messpunkte
bestätigen: 2.190 → 10.000 Zeilen ist Faktor 4,57, und 4,57² = 20,9 trifft
das Verhältnis der Speicherforderungen. Auf dieselbe Rechnung angewendet
heißt das für die Zeit:

    729 s × 20,9 ≈ 4,2 Stunden je Datensatz
    33 Datensätze    ≈ 5 bis 6 Tage

Das ist eine Hochrechnung aus **einer** Messung, nicht eine Messung — aber
auch bei linearer statt quadratischer Skalierung bleiben rund 30 Stunden. Auf
dieser Karte ist der Lauf nicht machbar.

**Ein Teil davon ist die Karte, nicht das Modell.** TabFM lädt seine Gewichte
in `bfloat16`. bfloat16-Tensorkerne gibt es erst ab Rechenfähigkeit 8.0
(Ampere: A100, L4, G4). Eine T4 ist 7.5 und hat Tensorkerne nur für
`float16`; PyTorch rechnet `bfloat16` dort trotzdem, nur ohne diese Kerne und
ohne Warnung. `torch.cuda.is_bf16_supported()` taugt nicht als Prüfung — der
belastbare Test ist `torch.cuda.get_device_capability()[0] >= 8`. Genau den
macht `run_models.py` jetzt beim Start und sagt es, wenn die Karte nicht
passt. Wie viel die fehlenden Kerne wirklich ausmachen, ist hier nicht
gemessen; es ist der erste Punkt, der auf einer Ampere-Karte zu prüfen ist.

#### Woran es genau liegt

Drei Dinge greifen ineinander. Alle drei stehen im Quelltext von TabFM, keines
in dessen README.

**1. Ein Fold ist eine einzige Folge.** TabFM lernt nicht, es liest die
Trainingsdaten als Kontext. Ein Fold geht deshalb als *eine* Folge durch das
Modell: 8.000 Kontextzeilen und 2.000 Testzeilen zusammen, 10.000 Positionen.
`_batch_forward()` zerlegt die Arbeit — aber nur über die Ensemble-Achse:

```python
batch_size_per_process = self.batch_size or Xs.shape[0]
n_batches = math.ceil(Xs.shape[0] / batch_size_per_process)
```

`Xs.shape[0]` ist die Zahl der Ensemble-Mitglieder. Die Folgenlänge wird nie
geteilt. `batch_size` ist also kein Speicherventil für große Datensätze,
sondern vereinzelt nur die 32 Mitglieder — und kostet dabei Zeit.

**2. Die Häppchengrößen sind für eine 40-GB-Karte gewählt.** Dagegen hat
TabFM intern durchaus etwas: Es zerlegt die großen Zwischenergebnisse selbst.
Die Größen stehen als Modulkonstanten in `pytorch/model.py`:

> Always-on activation-chunk sizes. TabFM runs the whole training fold as one
> in-context sequence, so a single forward materialises activations that grow
> with rows \* features and OOM the GPU on large tasks. […] Chunking is exact
> (identical outputs) […] and it costs <1% runtime otherwise […]
> **Sizes are chosen for memory safety on a 40 GB GPU** across TabArena-scale
> tasks.
>
> `_ROW_CHUNK_SIZE = 4096` · `_COL_CHUNK_SIZE = 16` · `_FFN_CHUNK_SIZE = 8192`

Auf einer Karte mit 14,56 GiB sind 4096er Häppchen zu groß. Das ist der
eigentliche Grund für die 33 Fehlschläge — nicht `batch_size`.

Die Autoren haben das vorgesehen. Im `TabFM`-Konstruktor steht über derselben
Schleife, die die Konstanten verteilt: *„The per-block knobs stay settable
(e.g. to None) for benchmarking."* `run_models.py` läuft deshalb nach dem
Laden einmal über `model.modules()` und setzt sie kleiner — dieselbe
Schleife, die TabFM selbst benutzt. Weil das Zerlegen laut Autoren **exakt**
ist, ändert es die Zahlen nicht.

**3. Die Aufmerksamkeit fällt auf den langsamsten Kern zurück.** `model.py`
rechnet sie so:

```python
o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
```

Die Maske ist entscheidend. PyTorchs **Flash**-Kern verträgt keine Maske,
fällt also von vornherein aus. Bleiben **mem_efficient** (Speicher wächst
linear mit der Folgenlänge) und **math** (rechnet die ganze Matrix
Folge × Folge aus, also quadratisch). Welcher von beiden geht, hängt von
Karte *und* Präzision ab — und für `bfloat16` verlangen die schnellen Kerne
Rechenfähigkeit 8.0. Auf einer T4 bleibt damit vermutlich nur `math`: die
quadratische Variante, in einer Präzision, für die die Karte keine
Tensorkerne hat.

„Vermutlich", weil das von der PyTorch-Version abhängt und ich es hier nicht
nachmessen kann. Deshalb misst `run_models.py` es jetzt selbst: Beim Start
probiert es jeden Kern einmal mit Maske in der gewählten Präzision aus und
schreibt hin, welcher geht. Das ersetzt die Vermutung durch eine Zeile im
Protokoll.

#### Was sich davon einstellen lässt

| Schalter | Wirkung aufs Ergebnis | Wofür |
|---|---|---|
| `--tabfm-chunk` | **keine** (laut Autoren exakt) | Speicher. Voreinstellung: nach Kartengröße, 1024 unter 20 GiB, 2048 unter 32, sonst 4096 |
| `--tabfm-batch` | **keine** (nur Bündelung) | Speicher, kostet aber Zeit. Zweite Wahl nach `--tabfm-chunk` |
| `--tabfm-dtype` | **ja** | Zeit. `bfloat16` (Voreinstellung), `float32`, `float16` |

Die Reihenfolge ist Absicht: Häppchen zuerst, weil sie laut Quelltext unter
1 % Laufzeit kosten; das Bündel erst danach, weil es die 32 Mitglieder
vereinzelt und damit echte Zeit kostet.

`--tabfm-dtype` ist der einzige der drei, der die Zahlen ändert, und steht
deshalb in `modell_version`. Was die `load()`-Doku dazu sagt:

> The checkpoint is stored in float32, but the model is designed to run in
> bfloat16 […] `dtype` is provided for float32 debugging / quality comparison

| | |
|---|---|
| `bfloat16` | wie vorgesehen. Tensorkerne dafür erst ab Rechenfähigkeit 8.0 |
| `float32` | die Präzision der gespeicherten Prüfsumme, von den Autoren zum Vergleich angeboten. Genauer als bfloat16, doppelter Speicher, auf der T4 ohne Nachbildung |
| `float16` | erwähnen die Autoren **nicht**. Nur dafür hat eine T4 Tensorkerne, aber der Wertebereich ist viel kleiner — Überlauf ist möglich |

#### Zwei Zeilen sagen `chunkauto`

In der Ergebnisdatei melden **S5E7** und **S4E3** `chunkauto`, alle anderen
TabFM-Zeilen `chunk2048`. Das ist ein Fehler im Protokoll, nicht in der
Rechnung: `modell_version` wurde früher am Anfang von `rechne()` gebildet,
und zu diesem Zeitpunkt hatte `modell()` die Gewichte noch nicht geladen —
die Häppchengröße stand also noch nicht fest. Betroffen war genau der
*erste* Datensatz eines Laufs; ab dem zweiten stand der Wert vom
vorhergehenden noch da. Gerechnet wurde in allen 33 Fällen mit 2048, denn
`modell()` setzt die Größe beim Laden, also vor der ersten Vorhersage.

Seit dem Fehler wird `modell_version` am Ende von `rechne()` noch einmal
gesetzt, wenn das Modell gebaut ist. Die beiden Zeilen bleiben so stehen, wie
sie gemessen wurden; wer sie neu rechnen will, braucht `--neu`.

Die Zeile zu **S5E3** stammt aus dem Lauf vom 14.11. und sagt nur
`tabfm v1.0.0 batch8` — damals gab es weder die Angabe der Präzision noch
die Häppchen. Gerechnet wurde sie in `bfloat16` (der Voreinstellung von
`load()`) mit TabFMs eigenen 4096er Häppchen, die bei 2.190 Zeilen auch auf
eine T4 passten. Weil das Zerlegen laut Autoren exakt ist, ist die Zeile mit
den anderen vergleichbar.

Wer die Präzision wechselt, muss **S5E3 neu rechnen**: Die vorhandene Zeile
ist in `bfloat16` entstanden (AUC 0,894161) und ist gleichzeitig die
Gegenprobe — weicht eine andere Präzision dort deutlich ab oder liefert
`nan`, ist sie unbrauchbar.

### TabDPT, das dritte Fundamentmodell

TabPFN und TabFM sind beide auf **synthetischen** Daten vortrainiert: Sie haben
nie eine echte Tabelle gesehen, sondern Millionen aus einem Prior erzeugte.
TabDPT ist das Gegenstück — es ist auf echten Tabellen trainiert und löst die
Größenbeschränkung über Retrieval. Als drittes Modell prüft es damit eine
andere Frage als ein weiteres synthetisches: ob die *Herkunft* der
Trainingsdaten überhaupt einen Unterschied macht.

Apache-2.0, `pip install tabdpt`, Gewichte von Hugging Face ohne Konto, und es
kann Klassifikation und Regression — Letzteres ist Pflicht, 12 der 34
Datensätze sind Regression.

TabDPT braucht keine Laufzeit für sich allein, sondern nur eine, in der
danach **kein `rf` und `xgb` mehr** gerechnet wird: Die Installation zieht
`torch`, `faiss` und `huggingface_hub` mit, und ob `scikit-learn` dabei auf
1.6.1 bleibt, ist nicht beobachtet. Die Auswertung aus Abschnitt 4 dieser
Anleitung darf dieselbe Laufzeit benutzen — `aggregate_ranks.py` importiert
kein `scikit-learn`, es liest nur die Ergebnisdatei.

#### Was dagegen spricht, und in die Limitationen gehört

Genau die Stärke ist auch das Risiko: **Kontamination.** Playground-Datensätze
sind synthetische Nachbildungen echter Datensätze, und die Vorlagen liegen
oft auf OpenML — also dort, wo TabDPT seine Trainingsdaten herhat. Es kann
Originale gesehen haben, aus denen unsere Daten erzeugt wurden. Das entwertet
den Vergleich nicht, aber es ist ein Konfundierer, und er muss benannt werden,
bevor jemand anders ihn benennt. Bei TabPFN und TabFM besteht dieses Risiko
nicht, und das ist beim Lesen der Ränge mitzudenken.

#### Die Hülle, und warum sie kein Tuning ist

`TabDPTHuelle` in `run_models.py` glättet vier Unterschiede. Keiner davon
verändert das Modell; sie stellen nur her, was die anderen vier Modelle von
sich aus können.

1. **Die Schalter sitzen auf `predict()`**, nicht im Konstruktor:
   `model.predict(X, n_ensembles=8, context_size=None, batch_size=None,
   seed=42)`. Eine `Pipeline` ruft `predict(X)` selbst auf und gibt nichts
   weiter, also werden die Werte in der Hülle gemerkt und beim Aufruf
   eingesetzt.

2. **`seed` ist `None` voreingestellt**, das Ensemble also nicht
   wiederholbar. Die Hülle setzt `SEED` — dasselbe, was `random_state=SEED`
   bei `rf`, `xgb` und TabPFN tut.

3. **Wahrscheinlichkeiten mit Ensemble heißen `ensemble_predict_proba()`.**
   `predict_proba()` kennt bei TabDPT gar kein `n_ensembles` und wäre ein
   einzelner Durchgang — also ein anderes Modell als das, das `predict()`
   benutzt. Das ist die unauffälligste Falle der ganzen Schnittstelle: Der
   Aufruf funktioniert, liefert plausible Zahlen und misst etwas anderes.
   Die Hülle ruft für beide Wege denselben auf.

4. **`fit()` besteht per `assert` auf numpy** („X must be a numpy array"),
   `y` kommt bei uns aber als `pandas.Series` aus dem Fold. Und `predict()`
   gibt bei der Klassifikation Klassen*indizes* zurück; ein `classes_`, das
   die Metriken brauchen, gibt es nicht. Die Hülle schiebt die Klassen je
   Fold auf 0..k-1 zusammen (`np.unique(..., return_inverse=True)`) und
   übersetzt zurück. Das Zusammenschieben ist nötig, weil TabDPT die Ausgabe
   auf die ersten `num_classes` Spalten abschneidet — fehlt in einem Fold
   eine seltene Klasse, zeigte Spalte 2 sonst auf die falsche.

#### Was sich einstellen lässt

| Schalter | Wirkung aufs Ergebnis | Wofür |
|---|---|---|
| `--tabdpt-haeppchen` | **keine** (nur Durchgänge) | Speicher. Voreinstellung der Bibliothek: 4096 auf der CPU, 128k auf der GPU |
| `--tabdpt-kontext` | **ja** | Speicher und Zeit. Voreinstellung: alle Zeilen |
| `--tabdpt-ensemble` | **ja** | Zeit. Voreinstellung: 8 Durchgänge |
| `--tabdpt-ohne-compile` | praktisch keine | Notausgang, falls `torch.compile` mit `InductorError` abbricht |

Bei Speichermangel verkleinert die Hauptschleife von selbst nur die
**Häppchen** — erst auf 8192, dann halbierend bis 256. Den Kontext fasst sie
nicht an, denn der würde das Ergebnis ändern. Das ist der Unterschied zu
TabFM, wo beide Stellschrauben ergebnisneutral sind.

`--tabdpt-kontext` verdient eine Erklärung, weil das TabDPT-Paper selbst mit
2048 rechnet, die Bibliothek aber `None` voreingestellt hat, also alle Zeilen.
Wir nehmen die Voreinstellung der Bibliothek, wie bei allen anderen Modellen
auch — die Studie vergleicht ungetunte Voreinstellungen, nicht die
Konfiguration aus dem jeweiligen Paper. Aus der Doku von `predict()`:

> `context_size`: Maximum number of train points in the context. Uses all
> points if `None`, which can lead to GPU OOMs.

Wer den Wert setzt, muss es in `modell_version` wiederfinden können — deshalb
steht er dort, und Zeilen mit verschiedenem Kontext dürfen nicht gemischt
werden.

### Was „ungetunt“ heißt

Gesetzt werden nur `random_state`, `n_jobs` und — bei den Fundamentmodellen —
`device`, die Bündelung und der Startwert. Nichts davon ist ein
Hyperparameter:

| Parameter | Zweck |
|---|---|
| `random_state` | macht den Lauf wiederholbar |
| `n_jobs` | nutzt die vorhandenen Kerne |
| `device` | legt TabFM auf die GPU, die es sonst nicht findet |
| `batch_size` | bündelt TabFMs Ensemble-Mitglieder je Durchgang |
| `row_chunk_size` / `ffn_chunk_size` | zerlegt TabFMs Zwischenergebnisse passend zur Karte |
| `seed` (TabDPT) | dasselbe wie `random_state`, nur an `predict()` übergeben |
| `batch_size` (TabDPT) | zerlegt die Testzeilen in Durchgänge |
| `verbose=False` (TabDPT) | nimmt den Fortschrittsbalken aus der Ausgabe |

Alles andere bleibt auf der Voreinstellung der Bibliothek.

`batch_size` verdient eine Erklärung, weil es nach einer Stellschraube
aussieht. TabFMs Voreinstellung ist **1**: Die 32 Ensemble-Mitglieder gehen
einzeln durch das Modell, laut Doku „Number of ensemble members to forward
through the model at once". Auf einer GPU sind das 32 winzige Aufrufe statt
eines großen, und die Karte wartet die meiste Zeit. Gerechnet wird in beiden
Fällen dasselbe, nur anders gebündelt — die Vorhersagen ändern sich nicht.
Dieselbe Art Entscheidung wie `n_jobs=-1` beim Wald. Umstellbar über
`--tabfm-batch`.

Bei „CUDA out of memory" halbiert die Hauptschleife den Wert für diesen
einen Datensatz und versucht es erneut, bis hinunter zu 1; welcher Wert es am
Ende war, steht in `modell_version` (`tabfm v1.0.0 batch4`). Das darf die
Schleife selbst entscheiden, weil sich am Ergebnis nichts ändert — und es
ist nötig, weil der Speicherbedarf vom Datensatz abhängt und alle
Datensätze in einem Lauf stehen.

Außerdem wird nach jedem Fehlschlag `torch.cuda.empty_cache()` gerufen.
PyTorch behält freigegebene Blöcke in einem eigenen Vorrat, statt sie dem
Treiber zurückzugeben; nach einem missglungenen Versuch über mehrere Gigabyte
ist dieser Vorrat zerstückelt, und der nächste Datensatz findet nicht einmal
mehr ein halbes Gigabyte am Stück. Im Lauf vom 14.11.2026 hat genau das
28 der 33 Fehlschläge verursacht: Sie scheiterten an **626 MiB** auf einer
Karte, auf der nach dem ersten Fehler nur noch 43 MiB am Stück frei waren.

Deshalb stehen `scikit-learn` und `xgboost` in `requirements.txt` mit `==`
statt `>=`: Die Voreinstellungen *sind* hier der Untersuchungsgegenstand.

Dass das kein theoretisches Risiko ist, zeigt eine Gegenprobe auf S5E3 — ein
Rechner, ein Python, nur die Bibliotheken getauscht:

| | Random Forest | XGBoost |
|---|---|---|
| `skl 1.6.1 / xgb 3.4.1` (gemessen) | 0.87947 | **0.866285** |
| `skl 1.7.2 / xgb 3.0.5` | 0.87947 | **0.863911** |

Beim Random Forest stimmen alle fünf Fold-Werte einzeln überein, und die
Voreinstellungen beider Versionen unterscheiden sich in keinem einzigen
Parameter — dort ist die Version gleichgültig. Bei XGBoost nicht. Gepinnt sind
deshalb die Versionen, unter denen gemessen wurde.

Nebenbei belegt dieselbe Gegenprobe die Reproduzierbarkeit: Der Colab-Lauf
(Python 3.13) ließ sich unter Windows mit Python 3.12 Fold für Fold exakt
nachrechnen.

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
`sekunden_gesamt`, `status`, `hinweis` und `umgebung`.

#### Welche der beiden Zeitspalten gilt

`sekunden_fit` misst nur `fit()`, aufsummiert über die fünf Folds.
`sekunden_gesamt` misst den ganzen Durchgang: Einlesen, Deckelung,
Vorverarbeitung, Training *und* Vorhersage.

Bei Wald und Boosting sind beide fast gleich — dort steckt die Arbeit im
Training. Bei den Fundamentmodellen **nicht**: TabPFN und TabFM trainieren
nicht, sie legen die Trainingsdaten als Kontext ab. Ihr `fit()` ist deshalb
fast kostenlos, und die gesamte Rechenzeit fällt in `predict_proba`, das
außerhalb dieses Zählers liegt. Ein TabFM-Datensatz, der in der
Fortschrittszeile mit „0,6 s" erscheint, hat in Wirklichkeit ein Vielfaches
davon gebraucht.

**Für jeden Laufzeitvergleich zwischen den Modellfamilien gilt also
`sekunden_gesamt`.** `sekunden_fit` ist nur innerhalb einer Familie
aussagekräftig — etwa Wald gegen Boosting, oder `n10000` gegen `voll`.

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
