"""Evaluation metrics: aspect-term F1 (exact, relaxed), aspect-sentiment pair F1, JSON validity,
bootstrap confidence intervals and the cross-domain drop.

Predictions and gold use the unified schema: a list of {"aspect", "sentiment", ...} per sentence.
Matching is one-to-one within a sentence. Relaxed match is TOKEN-level: same normalised tokens, one token
sequence is a contiguous part of the other ("battery" ~ "battery life"), or token Jaccard >= 0.5. It is never a
character substring test ("tea" does not match "steak").
"""
import random
import re
from typing import Callable, Dict, List


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def exact_match(a: str, b: str) -> bool:
    return norm(a) == norm(b)


def _contiguous(small: list, big: list) -> bool:
    """True if `small` occurs as a contiguous run of tokens inside `big`."""
    n = len(small)
    return any(big[i:i + n] == small for i in range(len(big) - n + 1))


def relaxed_match(a: str, b: str) -> bool:
    ta, tb = norm(a).split(), norm(b).split()
    if not ta or not tb:
        return False
    if ta == tb or _contiguous(ta, tb) or _contiguous(tb, ta):
        return True
    sa, sb = set(ta), set(tb)
    return len(sa & sb) / len(sa | sb) >= 0.5


def _count_matches(pred, gold, match: Callable, use_sentiment: bool) -> int:
    used, tp = set(), 0
    for p in pred:
        for j, g in enumerate(gold):
            if j in used:
                continue
            if match(p["aspect"], g["aspect"]) and (not use_sentiment or p["sentiment"] == g["sentiment"]):
                used.add(j)
                tp += 1
                break
    return tp


def counts(preds: List[list], golds: List[list], match, use_sentiment):
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        m = _count_matches(p, g, match, use_sentiment)
        tp += m
        fp += len(p) - m
        fn += len(g) - m
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def evaluate(preds: List[list], golds: List[list]) -> Dict[str, dict]:
    """Return P/R/F1 for aspect-term (exact, relaxed) and aspect-sentiment pair (exact, relaxed)."""
    return {
        "aspect_exact": prf(*counts(preds, golds, exact_match, False)),
        "aspect_relaxed": prf(*counts(preds, golds, relaxed_match, False)),
        "pair_exact": prf(*counts(preds, golds, exact_match, True)),
        "pair_relaxed": prf(*counts(preds, golds, relaxed_match, True)),
    }


def json_validity_rate(raw_outputs: List[str], validator: Callable[[str], bool]) -> float:
    return sum(1 for o in raw_outputs if validator(o)) / max(len(raw_outputs), 1)


def bootstrap_ci(preds, golds, metric=("pair_exact", "f1"), n_boot=1000, seed=42, alpha=0.05):
    """Bootstrap over sentences. Returns (point, low, high)."""
    rng = random.Random(seed)
    key, sub = metric
    n = len(preds)
    point = evaluate(preds, golds)[key][sub]
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(evaluate([preds[i] for i in idx], [golds[i] for i in idx])[key][sub])
    vals.sort()
    return point, vals[int(alpha / 2 * n_boot)], vals[int((1 - alpha / 2) * n_boot) - 1]


def cross_domain_drop(in_domain: float, target: float) -> dict:
    """In-domain minus target performance (absolute and relative)."""
    return {"in_domain": in_domain, "target": target, "abs_drop": in_domain - target,
            "rel_drop": (in_domain - target) / in_domain if in_domain else 0.0}
