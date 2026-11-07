import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent
overview = pd.read_csv(root / "metadata" / "playground_overview.csv", sep=";", encoding="utf-8-sig")
features = pd.read_csv(root / "metadata" / "dataset_features.csv", sep=";", encoding="utf-8-sig")
log = pd.read_csv(root / "logs" / "download_log.csv", sep=";", encoding="utf-8-sig")

log = log.drop_duplicates("slug", keep="last")
overview = overview.rename(columns={"slugs": "slug"})
basicdata = overview.merge(features, on="slug", how="left")
basicdata = basicdata.merge(log, on="slug", how="left")

basicdata.to_csv(root / "metadata" / "basicdata.csv", sep=";", index=False, encoding="utf-8-sig")