"""Convert annotated CSVs to gold labels (IDs + labels only, no text) and compute self-consistency.

Run:  python scripts/parse_annotations.py
Writes data/annotation/gold_labels.jsonl (publishable) and results/annotation_consistency.json
"""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import evaluate

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data/annotation"


def parse_cell(cell):
    out = []
    if isinstance(cell, str) and cell.strip():
        for part in cell.split(";"):
            if "|" in part:
                asp, sent = part.rsplit("|", 1)
                if asp.strip() and sent.strip().lower() in ("positive", "negative", "neutral"):
                    out.append({"aspect": asp.strip(), "sentiment": sent.strip().lower()})
    return out


b1 = pd.read_csv(ANN / "batch1.csv", dtype={"annotation": str})
done = b1[b1["annotation"].notna() & (b1["annotation"].astype(str).str.strip() != "")]
# reviews with no aspects are written as 'none' to separate them from not-yet-annotated rows
none_rows = b1[b1["annotation"].astype(str).str.strip().str.lower() == "none"]
labels = {r.review_id: parse_cell(r.annotation) for r in done.itertuples()}
for r in none_rows.itertuples():
    labels[r.review_id] = []
with open(ANN / "gold_labels.jsonl", "w") as f:
    for rid, aspects in labels.items():
        f.write(json.dumps({"review_id": rid, "aspects": aspects}) + "\n")
print("annotated reviews:", len(labels), "of", len(b1))

b2 = pd.read_csv(ANN / "batch2_recheck.csv", dtype={"annotation": str})
pairs = [(labels[r.review_id], parse_cell(r.annotation)) for r in b2.itertuples()
         if r.review_id in labels and isinstance(r.annotation, str) and r.annotation.strip()]
if pairs:
    res = evaluate([p[1] for p in pairs], [p[0] for p in pairs])
    (ROOT / "results/annotation_consistency.json").write_text(json.dumps({"n": len(pairs), **res}, indent=2))
    print("self-consistency over", len(pairs), "reviews: aspect F1 %.3f, pair F1 %.3f" % (
        res["aspect_exact"]["f1"], res["pair_exact"]["f1"]))
else:
    print("no recheck annotations yet")
