from collections import Counter
from src.canonical import Canonicaliser, review_categories, dominant, agreement, net_table, singular
from src.common import DATA, PROCESSED, read_jsonl
from src.eval_clothing import compute_scores

C = Canonicaliser()


def test_singular():
    assert [singular(w) for w in ("sizes", "straps", "dress", "stitches", "bodies", "wrinkles", "us")] == \
           ["size", "strap", "dress", "stitch", "body", "wrinkle", "us"]


def test_aspect_terms_and_synonyms():
    assert C.category("fabric") == C.category("Material") == "material"
    assert C.category("sizing") == "size" and C.category("Sizes") == "size"
    assert C.category("appearance") == C.category("design") == C.category("style") == "style"
    assert C.category("ease of use") == "usability"
    assert C.category("quality of the fabric") == "material"            # head term
    assert C.category("toe box") == "toe box"                           # unknown part keeps its own phrase


def test_generic_aspect_uses_reason_only_when_generic():
    assert C.explain("shoes", "comfortable") == ("comfort", "reason_term")
    assert C.explain("the dress", "runs small") == ("size", "reason_term")
    assert C.explain("it", "well made") == ("quality", "reason_term")
    assert C.explain("shoes", "great")[1] == "generic_unmapped"
    assert C.explain("sole", "comfortable") == ("sole", "fallback")     # a real part noun is not overridden by the reason


def test_review_merge_dominant_sentiment():
    assert dominant(Counter(positive=2, negative=1)) == "positive"
    assert dominant(Counter(positive=1, negative=1)) == "neutral"
    items = [{"aspect": "fabric", "sentiment": "negative", "reason": "thin"},
             {"aspect": "material", "sentiment": "negative", "reason": "cheap"},
             {"aspect": "fit", "sentiment": "positive", "reason": "great"},
             {"aspect": "fit", "sentiment": "negative", "reason": "tight"}]
    out = {a["aspect"]: a["sentiment"] for a in review_categories(items, C)}
    assert out == {"material": "negative", "fit": "neutral"}
    assert review_categories([{"aspect": "dress", "sentiment": "positive", "reason": "great"}], C, drop_unmapped_generic=True) == []


def test_agreement_perfect_and_inverted():
    gold = [[{"aspect": "fit", "sentiment": "positive"}, {"aspect": "price", "sentiment": "negative"}]] * 6 + \
           [[{"aspect": "fit", "sentiment": "negative"}, {"aspect": "price", "sentiment": "positive"}, {"aspect": "color", "sentiment": "positive"}]] * 6
    a = agreement(gold, gold, min_gold=5)
    assert a["categories_found_by_system"] == 3 and abs(a["net_sentiment_pearson"] - 1.0) < 1e-9
    inv = [[dict(x, sentiment="negative" if x["sentiment"] == "positive" else "positive") for x in r] for r in gold]
    assert agreement(gold, inv, min_gold=5)["net_sentiment_pearson"] < 0


def test_gold_labels_are_mostly_covered_by_the_taxonomy():
    gold = read_jsonl(DATA / "annotation" / "gold_labels.jsonl")
    routes = Counter(C.explain(a["aspect"])[1] for r in gold for a in r["aspects"])
    n = sum(routes.values())
    assert (routes["aspect_term"] + routes["head_term"]) / n > 0.7
    assert routes["generic_unmapped"] / n < 0.02


def test_compute_scores_oracle_and_failed_reviews():
    gold = {r["review_id"]: r["aspects"] for r in read_jsonl(DATA / "annotation" / "gold_labels.jsonl")}
    pipe = {r["review_id"]: r for r in read_jsonl(PROCESSED / "clothing_reviews.jsonl")}
    ids = [i for i in gold if i in pipe][:60]
    texts = [pipe[i]["text"] for i in ids]
    res = [{"items": [dict(a) for a in gold[i]], "ok": True, "n_sent": 1, "n_failed": 0} for i in ids]
    sc = compute_scores(res, [gold[i] for i in ids], texts, C)
    assert sc["metrics"]["pair_exact"]["f1"] == 1.0 and sc["scored_reviews"] == len(ids)
    res[0] = {"items": [], "ok": False, "n_sent": 1, "n_failed": 1}
    sc = compute_scores(res, [gold[i] for i in ids], texts, C)
    assert sc["excluded_reviews"] == 1 and sc["metrics_strict_all"]["pair_exact"]["f1"] < 1.0
    assert 0.3 < sc["gold_literal_span_share"] < 0.8
