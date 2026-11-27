"""Canonical aspect categories for the clothing evaluation and report.

Why: the clothing gold labels are mostly aspect CATEGORIES (comfort, style, durability ...), often inferred from
opinion words, while the extractors return literal SPANS from the review. Both sides are therefore mapped to the
same canonical categories (configs/canonical_aspects.json) before scoring and aggregation.

Mapping of an extracted item (aspect, reason):
  1. aspect phrase is a taxonomy term                      -> that category      (route "aspect_term")
  2. head (last) or first token of the phrase is a term     -> that category      (route "head_term")
  3. aspect is a generic/product word (dress, shoes, it):
       opinion words (reason) contain a taxonomy term       -> that category      (route "reason_term")
       otherwise                                            -> own phrase         (route "generic_unmapped")
  4. anything else keeps its own singularised phrase        -> (route "fallback")
Gold labels use the same function without a reason.
Per review, items are merged to ONE label per category (annotation guidelines 5 and 6): the dominant sentiment
wins, and a tie between positive and negative becomes neutral.
"""
import json, re
from collections import Counter
from .common import ROOT

CONFIG = ROOT / "configs" / "canonical_aspects.json"
_IRREGULAR = {"children": "child", "men": "man", "women": "woman", "feet": "foot", "teeth": "tooth"}


def singular(tok: str) -> str:
    if tok in _IRREGULAR:
        return _IRREGULAR[tok]
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith(("ches", "shes", "xes", "sses")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokens(s: str):
    return [singular(t) for t in re.findall(r"[a-z0-9]+", (s or "").lower().replace("'", ""))]


def _contiguous(small, big):
    n = len(small)
    return n > 0 and any(big[i:i + n] == small for i in range(len(big) - n + 1))


class Canonicaliser:
    def __init__(self, path=CONFIG):
        cfg = json.load(open(path, encoding="utf-8"))
        self.det = set(cfg["determiners"])
        self.generic = {t for g in cfg["generic_terms"] for t in [" ".join(tokens(g))]}
        self.aspect_idx, self.opinion_idx = {}, {}
        for cat, d in cfg["categories"].items():
            self.aspect_idx[cat] = cat                                   # a category name maps to itself
            for t in d["aspect_terms"]:
                self.aspect_idx[" ".join(tokens(t))] = cat
            for t in d["opinion_terms"]:
                self.opinion_idx[" ".join(tokens(t))] = cat
        self.aspect_idx.update({" ".join(tokens(c)): c for c in cfg["categories"]})
        self._opinion_keys = [(k.split(), c) for k, c in self.opinion_idx.items()]

    def _phrase(self, aspect):
        toks = [t for t in tokens(aspect) if t not in self.det]
        return toks

    def is_generic(self, aspect) -> bool:
        toks = self._phrase(aspect)
        return not toks or " ".join(toks) in self.generic or all(t in self.generic for t in toks)

    def explain(self, aspect, reason=""):
        """Returns (category, route)."""
        toks = self._phrase(aspect)
        key = " ".join(toks)
        if key in self.aspect_idx:
            return self.aspect_idx[key], "aspect_term"
        if toks:
            for t in (toks[-1], toks[0]):
                if t in self.aspect_idx and not (t in self.generic):
                    return self.aspect_idx[t], "head_term"
        if self.is_generic(aspect):
            rt, best = tokens(reason), None
            for ktoks, cat in self._opinion_keys:
                if _contiguous(ktoks, rt):
                    pos = next(i for i in range(len(rt)) if rt[i:i + len(ktoks)] == ktoks)
                    cand = (pos, -len(ktoks), cat)
                    best = cand if best is None or cand < best else best
            if best:
                return best[2], "reason_term"
            return key or "generic", "generic_unmapped"
        return key, "fallback"

    def category(self, aspect, reason="") -> str:
        return self.explain(aspect, reason)[0]


def dominant(votes: Counter) -> str:
    pos, neg = votes.get("positive", 0), votes.get("negative", 0)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def review_categories(items, canon, drop_unmapped_generic=False):
    """Merge a review's extracted items to one {"aspect": category, "sentiment"} per category."""
    votes = {}
    for it in items:
        cat, route = canon.explain(it["aspect"], it.get("reason", ""))
        if drop_unmapped_generic and route == "generic_unmapped":
            continue
        votes.setdefault(cat, Counter())[it["sentiment"]] += 1
    return [{"aspect": c, "sentiment": dominant(v)} for c, v in votes.items()]


def net_table(per_review):
    """Per-category counts over reviews: positive / negative / neutral / mentions (reviews) / net sentiment."""
    t = {}
    for items in per_review:
        for a in items:
            t.setdefault(a["aspect"], Counter())[a["sentiment"]] += 1
    out = {}
    for cat, c in t.items():
        m = sum(c.values())
        out[cat] = {"positive": c["positive"], "negative": c["negative"], "neutral": c["neutral"], "mentions": m,
                    "net_sentiment": (c["positive"] - c["negative"]) / m}
    return out


def _corr(xs, ys, method):
    import pandas as pd
    if len(xs) < 3:
        return None
    v = pd.Series(xs).corr(pd.Series(ys), method=method)
    return None if v != v else float(v)


def agreement(gold_per_review, sys_per_review, min_gold=10):
    """Proposal metric: correlation between system and gold net sentiment per category (categories with
    >= min_gold gold mentions; net sentiment needs >= 1 system mention). Also correlation of mention counts."""
    g, s = net_table(gold_per_review), net_table(sys_per_review)
    cats = sorted(c for c, v in g.items() if v["mentions"] >= min_gold)
    both = [c for c in cats if c in s]
    table = [{"category": c, "gold_mentions": g[c]["mentions"], "gold_net": g[c]["net_sentiment"],
              "sys_mentions": s[c]["mentions"] if c in s else 0,
              "sys_net": s[c]["net_sentiment"] if c in s else None} for c in cats]
    return {"min_gold_mentions": min_gold, "categories_in_gold": len(cats), "categories_found_by_system": len(both),
            "net_sentiment_pearson": _corr([g[c]["net_sentiment"] for c in both], [s[c]["net_sentiment"] for c in both], "pearson"),
            "net_sentiment_spearman": _corr([g[c]["net_sentiment"] for c in both], [s[c]["net_sentiment"] for c in both], "spearman"),
            "mention_count_spearman": _corr([g[c]["mentions"] for c in cats],
                                            [s[c]["mentions"] if c in s else 0 for c in cats], "spearman"),
            "table": table}
