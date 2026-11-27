"""Aggregate per-review extractions into per-aspect sentiment scores for the product-manager report.

v4: aspects are mapped to canonical categories (src/canonical.py, configs/canonical_aspects.json) using the
aspect AND the extracted opinion words, and each review contributes ONE vote per category (dominant sentiment),
the same unit as the gold labels. Generic/product words (dress, shoes, it) that cannot be mapped are dropped.
Net sentiment per category = (positive - negative) / mentions, where mentions = number of reviews.
The agreement with the gold labels (proposal metric) is computed in src/eval_clothing.py on the gold reviews.

Run:  python -m src.aggregate --method baseline            (or prompted / finetuned with --limit 300)
"""
import argparse
import pandas as pd
from .common import read_jsonl, RESULTS, PROCESSED
from .canonical import Canonicaliser, review_categories
from .extractors import get_extractor


def aggregate(items_per_review, min_mentions=5, canon=None) -> pd.DataFrame:
    """items_per_review: list (one entry per review) of extracted items with 'aspect', 'sentiment', 'reason'."""
    canon = canon or Canonicaliser()
    rows = [{"aspect": a["aspect"], "sentiment": a["sentiment"]}
            for items in items_per_review for a in review_categories(items, canon, drop_unmapped_generic=True)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    t = df.pivot_table(index="aspect", columns="sentiment", aggfunc="size", fill_value=0)
    for c in ("positive", "negative", "neutral"):
        if c not in t:
            t[c] = 0
    t["mentions"] = t[["positive", "negative", "neutral"]].sum(axis=1)
    t["net_sentiment"] = (t["positive"] - t["negative"]) / t["mentions"]
    return t[t["mentions"] >= min_mentions].sort_values("mentions", ascending=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="baseline", choices=["baseline", "baseline_lr", "prompted", "finetuned"])
    ap.add_argument("--input", default=str(PROCESSED / "clothing_reviews.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--min_mentions", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="number of reviews (LLM methods are slow)")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--adapter", default="adapters/absa-qlora")
    a = ap.parse_args()
    rows = read_jsonl(a.input)[: a.limit]
    sents = [s for r in rows for s in r["sentences"]]
    res = get_extractor(a.method, a.model, a.adapter)(sents)      # sentence-level extraction, then merged per review
    per_review, k = [], 0
    for r in rows:
        chunk = res[k:k + len(r["sentences"])]
        k += len(r["sentences"])
        per_review.append([it for x in chunk if x["ok"] for it in x["items"]])
    table = aggregate(per_review, a.min_mentions)
    out = a.out or str(RESULTS / f"aggregate_clothing_{a.method}.csv")
    table.to_csv(out)
    print(f"{len(rows)} reviews, {len(sents)} sentences, {sum(1 for r in res if r['ok'])} valid extractions -> {out}")
    print(table.head(15).round(2).to_string())
