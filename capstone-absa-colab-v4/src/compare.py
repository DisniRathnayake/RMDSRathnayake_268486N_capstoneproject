"""Collect all result files into one comparison table (results/comparison.md).

Primary clothing metrics are CATEGORY level (see src/eval_clothing.py). The drop is computed on the SPAN level
(source span F1 vs clothing span F1 on gold that literally occurs in the review), the only like-for-like pair.

Run:  python -m src.compare
"""
import json
from .common import RESULTS


def load(p):
    try:
        return json.load(open(RESULTS / p))
    except FileNotFoundError:
        return None


def f(x, spec="%.3f"):
    return spec % x if x is not None else "n/a"


def main():
    head = ["Method", "Source test pair F1 (span)", "Clothing pair F1 (category)", "Clothing aspect F1 (category)",
            "Clothing pair F1 (category, strict all reviews)", "Clothing pair F1 (span, literal gold)",
            "Abs. drop (span)", "Rel. drop (span)", "Net-sentiment corr. (Spearman)", "Scored / gold reviews",
            "JSON validity (clothing sentences)", "s/review (clothing)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in ("baseline", "baseline_lr", "prompted", "finetuned", "finetuned+setfit"):
        c = load(f"clothing_{m}.json")
        if not c:
            continue
        d = c["cross_domain_drop_span"].get("pair_exact")
        lines.append("| " + " | ".join([
            m, f(d["in_domain"]) if d else "n/a", f(c["metrics"]["pair_exact"]["f1"]),
            f(c["metrics"]["aspect_exact"]["f1"]), f(c["metrics_strict_all"]["pair_exact"]["f1"]),
            f(c["metrics_span_literal_gold"]["pair_exact"]["f1"]),
            f(d["abs_drop"]) if d else "n/a", ("%.1f%%" % (100 * d["rel_drop"])) if d else "n/a",
            f(c["agreement"]["net_sentiment_spearman"]), "%d / %d" % (c["scored_reviews"], c["gold_reviews"]),
            f(c["sentence_json_validity_rate"]), f(c["seconds_per_review"], "%.2f")]) + " |")
    ref = load("clothing_baseline.json") or next((load(f"clothing_{m}.json") for m in ("prompted", "finetuned") if load(f"clothing_{m}.json")), None)
    if ref:
        lines += ["", "Gold aspects that occur literally in the review: %.1f%% (whole word %.1f%%). The span-level column can only score that share; the category-level columns score all gold aspects." % (
            100 * ref["gold_literal_span_share"], 100 * ref["gold_literal_whole_word_share"])]
    (RESULTS / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
