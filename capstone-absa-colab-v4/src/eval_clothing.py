"""Evaluate an extractor on the clothing gold set (target domain, no target training).

Protocol (v4):
 * Text and sentence boundaries come from data/processed/clothing_reviews.jsonl, the same cleaned text the
   pipeline and aggregation use (the gold sheet's raw text is no longer read).
 * Each sentence is extracted, items are merged per review.
 * PRIMARY metric = CATEGORY level. The gold labels are mostly aspect categories (comfort, style, ...), the
   extractors return review spans, so both are mapped to canonical categories (src/canonical.py,
   configs/canonical_aspects.json) and merged to one label per category per review (dominant sentiment).
 * SECONDARY metric = SPAN level on the part of the gold that literally occurs in the review
   (like-for-like with the source-domain span F1, used for the cross-domain drop).
 * AGREEMENT = correlation between system and gold net sentiment per category (proposal metric).
 * A review is excluded from `metrics` if any sentence produced invalid LLM output; `metrics_strict_all`
   counts such reviews with the items that were produced (no survivorship bias).

Run:  python -m src.eval_clothing --method baseline
      python -m src.eval_clothing --method prompted --model meta-llama/Meta-Llama-3-8B-Instruct
      python -m src.eval_clothing --method finetuned --adapter adapters/absa-qlora
"""
import argparse, json, re, time
from collections import Counter
from .common import DATA, PROCESSED, RESULTS, read_jsonl, write_jsonl, set_seed
from .canonical import Canonicaliser, review_categories, agreement
from .extractors import get_extractor, extract_reviews
from .metrics import evaluate, bootstrap_ci, cross_domain_drop


def source_f1(method, key="pair_exact"):
    """In-domain (source test) span F1 for the same method, from results/*.json."""
    try:
        if method in ("baseline", "baseline_lr"):
            d = json.load(open(RESULTS / "baseline_test.json"))
            return d["spacy_lexicon" if method == "baseline" else "spacy_tfidf_lr"]["all"][key]["f1"]
        return json.load(open(RESULTS / f"{method}_test.json"))["metrics"][key]["f1"]
    except FileNotFoundError:
        return None


def _in_text(aspect, text, whole_word):
    a, t = aspect.lower(), text.lower()
    if whole_word:
        return re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", t) is not None
    return a in t


def compute_scores(res, gold_items, texts, canon):
    """res: extract_reviews output; gold_items: per-review gold [{aspect, sentiment}]; texts: review texts."""
    keep = [i for i, r in enumerate(res) if r["ok"]]
    gold_cat = [review_categories(g, canon) for g in gold_items]
    sys_cat = [review_categories(r["items"], canon) for r in res]
    out = {"metrics": evaluate([sys_cat[i] for i in keep], [gold_cat[i] for i in keep]),
           "metrics_strict_all": evaluate(sys_cat, gold_cat),
           "scored_reviews": len(keep), "excluded_reviews": len(res) - len(keep)}
    if len(keep) >= 10:
        out["pair_exact_f1_ci"] = bootstrap_ci([sys_cat[i] for i in keep], [gold_cat[i] for i in keep], n_boot=500)
        out["aspect_exact_f1_ci"] = bootstrap_ci([sys_cat[i] for i in keep], [gold_cat[i] for i in keep],
                                                 metric=("aspect_exact", "f1"), n_boot=500)
    out["agreement"] = agreement(gold_cat, sys_cat)
    # span level, gold restricted to aspects that literally occur in the review (substring, as the validator requires)
    lit = [[a for a in g if _in_text(a["aspect"], t, False)] for g, t in zip(gold_items, texts)]
    n_g = sum(len(g) for g in gold_items)
    out["gold_aspects"] = n_g
    out["gold_literal_span_share"] = sum(len(g) for g in lit) / max(n_g, 1)
    out["gold_literal_whole_word_share"] = sum(1 for g, t in zip(gold_items, texts) for a in g
                                               if _in_text(a["aspect"], t, True)) / max(n_g, 1)
    out["metrics_span_literal_gold"] = evaluate([r["items"] for r in res], lit)
    out["avg_predictions_per_review"] = sum(len(r["items"]) for r in res) / max(len(res), 1)
    routes = Counter(canon.explain(a["aspect"], a.get("reason", ""))[1] for r in res for a in r["items"])
    out["prediction_mapping_routes"] = dict(routes)
    out["gold_mapping_routes"] = dict(Counter(canon.explain(a["aspect"])[1] for g in gold_items for a in g))
    out["_gold_cat"], out["_sys_cat"] = gold_cat, sys_cat
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["baseline", "baseline_lr", "prompted", "finetuned"])
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--adapter", default="adapters/absa-qlora")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. +setfit")
    a = ap.parse_args()
    set_seed()
    gold = {r["review_id"]: r["aspects"] for r in read_jsonl(DATA / "annotation" / "gold_labels.jsonl")}
    pipeline = {r["review_id"]: r for r in read_jsonl(PROCESSED / "clothing_reviews.jsonl")}
    ids = [rid for rid in gold if rid in pipeline]
    if len(ids) < len(gold):
        print(f"WARNING: {len(gold) - len(ids)} gold reviews are not in the pipeline data and are skipped")
    texts = [pipeline[i]["text"] for i in ids]
    sentences = {pipeline[i]["text"]: pipeline[i]["sentences"] for i in ids}     # same units as the aggregation
    print("gold reviews:", len(ids))
    extractor = get_extractor(a.method, a.model, a.adapter)
    t0 = time.time()
    res = extract_reviews(texts, extractor, sentences.__getitem__)
    secs = (time.time() - t0) / max(len(ids), 1)
    n_sent, n_fail = sum(r["n_sent"] for r in res), sum(r["n_failed"] for r in res)
    canon = Canonicaliser()
    sc = compute_scores(res, [gold[i] for i in ids], texts, canon)
    gold_cat, sys_cat = sc.pop("_gold_cat"), sc.pop("_sys_cat")
    out = {"method": a.method + a.tag, "domain": "clothing", "gold_reviews": len(ids),
           "sentence_json_validity_rate": 1 - n_fail / max(n_sent, 1), "seconds_per_review": secs,
           "level_of_primary_metrics": "category (canonical aspect categories, one label per category per review)", **sc}
    write_jsonl([{"review_id": rid, "ok": r["ok"], "pred": r["items"], "gold": gold[rid],
                  "pred_categories": p, "gold_categories": g}
                 for rid, r, p, g in zip(ids, res, sys_cat, gold_cat)], RESULTS / f"preds_clothing_{a.method}{a.tag}.jsonl")
    # like-for-like drop: source span F1 vs clothing span F1 on literal gold. The category drop is informational only
    # (source is span level, clothing category level).
    out["cross_domain_drop_span"] = {}
    out["cross_domain_drop_category_informational"] = {}
    for key in ("aspect_exact", "pair_exact"):
        s = source_f1(a.method, key)
        if s is not None:
            out["cross_domain_drop_span"][key] = cross_domain_drop(s, out["metrics_span_literal_gold"][key]["f1"])
            out["cross_domain_drop_category_informational"][key] = cross_domain_drop(s, out["metrics"][key]["f1"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"clothing_{a.method}{a.tag}.json").write_text(json.dumps(out, indent=2))
    m, ms, ag = out["metrics"], out["metrics_span_literal_gold"], out["agreement"]
    print("%s on clothing | scored %d/%d | CATEGORY aspect F1 %.3f pair F1 %.3f | SPAN (literal gold) pair F1 %.3f | %.2fs/review" % (
        a.method, out["scored_reviews"], len(ids), m["aspect_exact"]["f1"], m["pair_exact"]["f1"], ms["pair_exact"]["f1"], secs))
    print("  net-sentiment agreement over %d categories: Pearson %s Spearman %s" % (
        ag["categories_found_by_system"], ag["net_sentiment_pearson"], ag["net_sentiment_spearman"]))
    for k, v in out["cross_domain_drop_span"].items():
        print("  span drop", k, "source %.3f -> clothing %.3f (abs %.3f, rel %.1f%%)" % (
            v["in_domain"], v["target"], v["abs_drop"], 100 * v["rel_drop"]))


if __name__ == "__main__":
    main()
