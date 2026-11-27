"""Preprocess the clothing target data (Amazon Reviews 2023 sample): cleaning, deduplication, sentence splitting.

Output data/processed/clothing_reviews.jsonl (contains review text, so it is git-ignored) and
results/clothing_preprocessing_stats.json (counts only, publishable).
Reviewer identifiers are never kept: only review_id (hash), product id, rating, title and text.

Run:  python -m src.prepare_clothing
"""
import html, json, re
import spacy
from .common import DATA, PROCESSED, RESULTS, read_jsonl, write_jsonl

SRC = DATA / "raw" / "amazon_clothing_sample.jsonl"
KEEP = ("review_id", "parent_asin", "rating", "title", "text")
MIN_WORDS = 5


def clean(t: str) -> str:
    t = html.unescape(t or "")
    t = re.sub(r"<[^>]+>", " ", t)                   # html tags
    t = re.sub(r"https?://\S+", " ", t)              # urls
    return re.sub(r"\s+", " ", t).strip()


def main():
    rows = read_jsonl(SRC)
    stats = {"raw_reviews": len(rows)}
    leaked = {k for r in rows for k in r if "user" in k.lower() or "reviewer" in k.lower()}
    stats["identifier_fields_found_and_dropped"] = sorted(leaked)
    out, seen = [], set()
    for r in rows:
        text = clean(r.get("text"))
        if len(text.split()) < MIN_WORDS:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({**{k: r.get(k) for k in KEEP}, "text": text, "title": clean(r.get("title"))})
    stats["after_cleaning_dedup"] = len(out)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])     # keeps the parser for sentence boundaries
    for r, doc in zip(out, nlp.pipe([r["text"] for r in out], batch_size=128)):
        r["sentences"] = [s.text.strip() for s in doc.sents if len(s.text.split()) >= 1]
    n_sent = [len(r["sentences"]) for r in out]
    words = [len(s.split()) for r in out for s in r["sentences"]]
    stats.update({"sentences_total": sum(n_sent), "sentences_per_review_mean": round(sum(n_sent) / len(out), 2),
                  "sentence_length_words_mean": round(sum(words) / len(words), 2)})
    write_jsonl(out, PROCESSED / "clothing_reviews.jsonl")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "clothing_preprocessing_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
