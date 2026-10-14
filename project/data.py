"""Parse the corpus and remove exact normalized-text overlap before splitting."""
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .common import ROOT, download_member

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def obtain_data():
    return download_member(URL, ROOT / "data/raw/SMSSpamCollection", "SMSSpamCollection")

def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()

def load_data(path, mode="full"):
    rows = []
    for index, line in enumerate(__import__("pathlib").Path(path).read_text(encoding="utf-8-sig").splitlines()):
        if not line.strip():
            continue
        if "\t" not in line:
            raise ValueError(f"Row {index + 1} has no label-text tab separator.")
        label, text = line.split("\t", 1)
        if label not in ("ham", "spam") or not text.strip():
            raise ValueError(f"Invalid SMS label or empty text at row {index + 1}.")
        rows.append({"source_row": index, "text": text, "y": int(label == "spam"),
                     "normalized": normalize(text)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("The corpus contains no messages.")
    original = len(frame)
    conflicting = set(frame.groupby("normalized")["y"].nunique().loc[lambda x: x > 1].index)
    conflict_rows = int(frame.normalized.isin(conflicting).sum())
    frame = frame.loc[~frame.normalized.isin(conflicting)]
    before_deduplication = len(frame)
    frame = frame.drop_duplicates("normalized", keep="first")
    deduplicated = len(frame)
    if mode == "quick" and len(frame) > 2200:
        sample, _ = train_test_split(np.arange(len(frame)), train_size=2200,
                                    random_state=42, stratify=frame.y)
        frame = frame.iloc[np.sort(sample)]
    frame = frame.reset_index(drop=True)
    if len(frame) < 100 or frame.y.value_counts().min() < 20:
        raise ValueError("Use at least 100 unique messages with at least 20 per class.")
    audit = {"original_rows": original, "conflicting_label_rows_removed": conflict_rows,
             "duplicates_removed": before_deduplication - deduplicated,
             "rows_after_deduplication": deduplicated, "rows_used": len(frame)}
    return frame, audit

def partition(frame, seed=42):
    ids = np.arange(len(frame))
    rest, test = train_test_split(ids, test_size=.20, stratify=frame.y, random_state=seed)
    train, validation = train_test_split(rest, test_size=.25, stratify=frame.y.iloc[rest],
                                        random_state=seed + 1)
    return {"train": train, "validation": validation, "test": test}
