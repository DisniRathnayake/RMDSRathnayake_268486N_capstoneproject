"""Non-LLM baselines.

1. SpacyLexiconExtractor: noun candidates found via dependency patterns (amod, acomp, conj, neg),
   sentiment from the VADER lexicon. Only nouns attached to an opinion word are emitted.
2. TfidfLRSentiment: classical-ML aspect sentiment classifier (TF-IDF over a context window around
   the aspect + logistic regression), trained on source-domain data only.

Run:  python -m src.baseline_spacy --split dev
"""
import argparse, json
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .common import PROCESSED, RESULTS, read_jsonl, set_seed
from .metrics import evaluate, bootstrap_ci
from .textnorm import ctx_tokens

NLP = None
VADER = SentimentIntensityAnalyzer()
POS_T, NEG_T = 0.25, -0.25


def nlp():
    global NLP
    if NLP is None:
        NLP = spacy.load("en_core_web_sm")
    return NLP


def word_score(tok):
    s = VADER.lexicon.get(tok.lemma_.lower(), VADER.lexicon.get(tok.text.lower(), 0.0)) / 4.0
    if any(c.dep_ == "neg" for c in tok.children):
        s = -s
    return s


def label(score):
    return "positive" if score >= POS_T else "negative" if score <= NEG_T else None


def noun_phrase(tok):
    parts = [c for c in tok.lefts if c.dep_ in ("compound", "amod") and c.pos_ in ("NOUN", "PROPN")]
    return " ".join([c.text for c in parts] + [tok.text]).lower()


def extract(doc):
    found = {}
    for tok in doc:
        if tok.pos_ not in ("NOUN", "PROPN") or tok.dep_ == "compound":
            continue
        ops = [c for c in tok.children if c.dep_ == "amod" and c.pos_ == "ADJ"]
        if tok.dep_ in ("nsubj", "nsubjpass") and tok.head.pos_ in ("AUX", "VERB", "ADJ"):
            ops += [c for c in tok.head.children if c.dep_ in ("acomp", "attr", "oprd")]
            if tok.head.pos_ == "ADJ":
                ops.append(tok.head)
        if tok.dep_ in ("dobj", "obj") and tok.head.pos_ == "VERB":
            ops.append(tok.head)                       # "love the screen"
        for o in list(ops):
            ops += [c for c in o.children if c.dep_ == "conj"]
        scored = [(o, word_score(o)) for o in ops if word_score(o) != 0.0]
        if not scored:
            continue
        o, s = max(scored, key=lambda x: abs(x[1]))
        lab = label(s)
        if lab:
            found[noun_phrase(tok)] = {"aspect": noun_phrase(tok), "sentiment": lab, "reason": o.text}
    return list(found.values())


def predict(texts):
    return [extract(d) for d in nlp().pipe(texts, batch_size=64)]


class TfidfLRSentiment:
    """Context-window sentiment classifier for a given aspect term."""
    def __init__(self, window=6):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.w = window
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")

    def _ctx(self, text, aspect):
        toks = ctx_tokens(text)                # punctuation-aware, so "life." still matches the aspect "life"
        a = ctx_tokens(aspect)
        for i in range(len(toks)):
            if toks[i:i + len(a)] == a:
                return " ".join(toks[max(0, i - self.w): i + len(a) + self.w])
        return " ".join(toks)

    def fit(self, rows):
        X, y = [], []
        for r in rows:
            for a in r["aspects"]:
                X.append(self._ctx(r["text"], a["aspect"]))
                y.append(a["sentiment"])
        self.clf.fit(self.vec.fit_transform(X), y)
        return self

    def predict(self, text, aspect):
        return self.clf.predict(self.vec.transform([self._ctx(text, aspect)]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    set_seed()
    rows = read_jsonl(PROCESSED / f"{args.split}.jsonl")
    golds = [r["aspects"] for r in rows]
    out = {"split": args.split, "n_sentences": len(rows)}
    preds = predict([r["text"] for r in rows])
    out["spacy_lexicon"] = {d: evaluate([p for p, r in zip(preds, rows) if r["domain"] == d],
                                        [g for g, r in zip(golds, rows) if r["domain"] == d])
                            for d in ("laptops", "restaurants")}
    out["spacy_lexicon"]["all"] = evaluate(preds, golds)
    out["spacy_lexicon"]["pair_exact_f1_ci"] = bootstrap_ci(preds, golds, n_boot=300)
    # classical alternative: spaCy extraction + TF-IDF/LR sentiment head
    lr = TfidfLRSentiment().fit(read_jsonl(PROCESSED / "train.jsonl"))
    preds_lr = [[dict(a, sentiment=lr.predict(r["text"], a["aspect"])) for a in p]
                for p, r in zip(preds, rows)]
    out["spacy_tfidf_lr"] = {"all": evaluate(preds_lr, golds)}
    # sentiment-only accuracy with gold aspects
    gold_pairs = [(r["text"], a) for r in rows for a in r["aspects"]]
    acc = sum(lr.predict(t, a["aspect"]) == a["sentiment"] for t, a in gold_pairs) / len(gold_pairs)
    out["tfidf_lr_sentiment_accuracy_gold_aspects"] = acc
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"baseline_{args.split}.json").write_text(json.dumps(out, indent=2))
    for k in ("spacy_lexicon", "spacy_tfidf_lr"):
        e = out[k]["all"]
        print(k, "aspect F1 exact %.3f relaxed %.3f | pair F1 exact %.3f relaxed %.3f" % (
            e["aspect_exact"]["f1"], e["aspect_relaxed"]["f1"], e["pair_exact"]["f1"], e["pair_relaxed"]["f1"]))
    print("TF-IDF/LR sentiment acc (gold aspects): %.3f" % acc)


if __name__ == "__main__":
    main()
