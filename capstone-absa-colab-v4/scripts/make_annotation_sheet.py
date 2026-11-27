"""Create the clothing gold-set annotation sheets.

Batch 1: N reviews stratified by star rating (equal counts per rating where possible).
Batch 2: a random 10% of batch 1, shuffled, to re-annotate AFTER A GAP for the self-consistency check.
Sheets contain review text, so they are git-ignored. Only labels + review IDs are published.

Annotate in the `annotation` column as:  aspect|sentiment; aspect|sentiment   (leave empty if none)
sentiment is positive, negative or neutral. The aspect must be copied from the review.

Run:  python scripts/make_annotation_sheet.py --n 300
"""
import argparse, random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--input", default=None)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--recheck_frac", type=float, default=0.10)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

cands = [ROOT / "data/raw/amazon_clothing_sample.jsonl", ROOT / "data/raw/clothing_fallback_womens.jsonl"]
src = Path(a.input) if a.input else next(p for p in cands if p.exists())
df = pd.read_json(src, lines=True)
df = df[df["text"].str.split().str.len().between(15, 120)]       # annotatable length
per = a.n // 5
parts = [g.sample(min(per, len(g)), random_state=a.seed) for _, g in df.groupby("rating")]
b1 = pd.concat(parts).sample(frac=1, random_state=a.seed).reset_index(drop=True)
b1["annotation"] = ""
out = ROOT / "data/annotation"
out.mkdir(parents=True, exist_ok=True)
b1[["review_id", "rating", "text", "annotation"]].to_csv(out / "batch1.csv", index=False)
b2 = b1.sample(frac=a.recheck_frac, random_state=a.seed + 1)[["review_id", "rating", "text"]].copy()
b2["annotation"] = ""
b2.to_csv(out / "batch2_recheck.csv", index=False)
print(f"source {src.name}: batch1 {len(b1)} reviews, batch2 recheck {len(b2)} reviews")
print(b1["rating"].value_counts().sort_index().to_dict())
