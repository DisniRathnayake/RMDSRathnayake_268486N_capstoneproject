"""One interface for all extractors, so evaluation, aggregation and the demo share the same code.

get_extractor(method) returns f(sentences) -> list of {"items": [...], "ok": bool}
Methods: baseline (spaCy + VADER), baseline_lr (spaCy + TF-IDF/LR sentiment), prompted, finetuned.
"""
from .common import PROCESSED, read_jsonl


def get_extractor(method, model="meta-llama/Meta-Llama-3-8B-Instruct", adapter="adapters/absa-qlora"):
    if method in ("baseline", "baseline_lr"):
        from . import baseline_spacy as bl
        lr = bl.TfidfLRSentiment().fit(read_jsonl(PROCESSED / "train.jsonl")) if method == "baseline_lr" else None

        def f(sentences):
            out = []
            for items, s in zip(bl.predict(sentences), sentences):
                if lr:
                    items = [dict(a, sentiment=lr.predict(s, a["aspect"])) for a in items]
                out.append({"items": items, "ok": True})
            return out
        return f
    if method in ("prompted", "finetuned"):
        from .llm_prompted import hf_generator, extract_with_retry
        gen = hf_generator(model, adapter if method == "finetuned" else None)

        def f(sentences):
            res = [extract_with_retry(s, gen, few_shot=(method == "prompted")) for s in sentences]
            return [{"items": r["items"], "ok": r["ok"]} for r in res]
        return f
    raise ValueError(method)


def extract_reviews(reviews, extractor, sentence_splitter):
    """reviews: list of text. Returns per review {"items", "ok", "n_sent", "n_failed"} (items merged over sentences)."""
    out = []
    for text in reviews:
        sents = [s for s in sentence_splitter(text) if s.strip()]
        res = extractor(sents)
        seen, items = set(), []
        for r in res:
            for a in r["items"]:
                k = (a["aspect"].lower(), a["sentiment"])
                if k not in seen:
                    seen.add(k)
                    items.append(a)
        failed = sum(1 for r in res if not r["ok"])
        out.append({"items": items, "ok": failed == 0, "n_sent": len(sents), "n_failed": failed})
    return out


def spacy_splitter():
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return lambda t: [s.text for s in nlp(t).sents]
