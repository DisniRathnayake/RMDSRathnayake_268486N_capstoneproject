"""Build the unified source-domain datasets.

Splits come from ASTE-Data-V2 (official train/dev/test for SemEval-2014 laptops and restaurants).

Fixes applied here (v4):
 1. DETOKENISATION. ASTE text is PTB-tokenised ("battery life .", "do n't"); the clothing reviews are natural
    text. Sentences are detokenised and the aspect/opinion spans are cut from the detokenised string, so
    train, dev, test and clothing all use the same text style.
 2. DEDUPLICATION. ASTE has one row per (aspect, opinion) pair, so an aspect with two opinions appears twice.
    Entries are merged by (normalised aspect, sentiment). `reason` keeps the first opinion span (it must stay a
    span of the sentence for the output validator); `reasons` lists all of them.
 3. ASPECT-FREE DEV/TEST. ASTE keeps only sentences with an aspect-opinion pair. Sentences with NO aspect term
    are taken from the SemEval-2014 train XML and split into disjoint train / dev / test portions, with the
    same ratio in every split, so source F1 also penalises false positives on aspect-free text.
    (The XML files in data/external are the TRAIN files, so the test portion is not from the official test XML.)

The 'reason' field is the annotated opinion span (a span of the review text).

Run:  python -m src.prepare_data
"""
import ast, json, random, xml.etree.ElementTree as ET
from collections import Counter
from .common import DATA, PROCESSED, RESULTS, SEED, write_jsonl
from .metrics import norm
from .textnorm import detokenize

ASTE = DATA / "external" / "aste_v2"
SEMEVAL = DATA / "external" / "semeval2014"
DOMAINS = {"14lap": "laptops", "14res": "restaurants"}
SPLITS = ("train", "dev", "test")
POL = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}
NO_ASPECT_RATIO = 0.25   # aspect-free sentences as a fraction of aspect-bearing sentences, in EVERY split


def load_aste(code, domain, split):
    """Returns (rows, stats). Rows are detokenised and deduplicated by (aspect, sentiment)."""
    rows, st = [], Counter()
    for i, line in enumerate(open(ASTE / code / f"{split}_triplets.txt", encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        raw, trip = line.split("####")
        toks = raw.split(" ")
        text, spans = detokenize(toks)
        st["sentences"] += 1
        st["sentences_changed_by_detok"] += text != raw
        merged = {}                                           # (norm aspect, sentiment) -> entry, insertion-ordered
        for a, o, p in ast.literal_eval(trip):
            asp = text[spans[a[0]][0]:spans[a[-1]][1]]
            opi = text[spans[o[0]][0]:spans[o[-1]][1]]
            st["aspect_entries_raw"] += 1
            key = (norm(asp) or asp.lower(), POL[p])
            if key not in merged:
                merged[key] = {"aspect": asp, "sentiment": POL[p], "reason": opi, "reasons": [opi]}
            elif opi not in merged[key]["reasons"]:
                merged[key]["reasons"].append(opi)
        st["aspect_entries_dedup"] += len(merged)
        rows.append({"id": f"{domain}_{split}_{i}", "domain": domain, "split": split,
                     "text": text, "aspects": list(merged.values())})
    return rows, st


def load_no_aspect_pool(domain):
    """Unique SemEval-2014 train sentences that have no aspect term (natural text)."""
    root = ET.parse(SEMEVAL / f"{domain}_train.xml").getroot()
    pool, seen = [], set()
    for s in root.iter("sentence"):
        if next(s.iter("aspectTerm"), None) is not None:
            continue
        t = " ".join((s.find("text").text or "").split())
        if t and t.lower() not in seen:
            seen.add(t.lower())
            pool.append(t)
    return pool


def main():
    rng = random.Random(SEED)
    by = {}                                                    # (domain, split) -> aspect-bearing rows
    stats = {"detokenisation_and_dedup": {}}
    for code, domain in DOMAINS.items():
        for split in SPLITS:
            by[(domain, split)], st = load_aste(code, domain, split)
            stats["detokenisation_and_dedup"][f"{domain}_{split}"] = dict(st)

    # train: drop duplicate sentences and sentences that also occur in dev/test (leakage)
    held = {r["text"].lower() for (d, sp), rows in by.items() if sp != "train" for r in rows}
    n_train_before = sum(len(by[(d, "train")]) for d in DOMAINS.values())
    seen = set()
    for domain in DOMAINS.values():
        kept = []
        for r in by[(domain, "train")]:
            k = r["text"].lower()
            if k not in held and k not in seen:
                seen.add(k)
                kept.append(r)
        by[(domain, "train")] = kept
    removed = n_train_before - sum(len(by[(d, "train")]) for d in DOMAINS.values())
    print("train sentences removed (duplicates / leakage):", removed)
    stats["train_sentences_removed_duplicates_or_leakage"] = removed

    # aspect-free sentences: one disjoint slice per split, same ratio everywhere
    used = {r["text"].lower() for rows in by.values() for r in rows}
    out = {sp: [] for sp in SPLITS}
    stats["no_aspect"] = {}
    for domain in DOMAINS.values():
        pool = [t for t in load_no_aspect_pool(domain) if t.lower() not in used]
        rng.shuffle(pool)
        start = 0
        for split in SPLITS:
            n = round(NO_ASPECT_RATIO * len(by[(domain, split)]))
            picked = pool[start:start + n]
            start += n
            used.update(t.lower() for t in picked)
            out[split] += by[(domain, split)] + [
                {"id": f"{domain}_{split}_noasp_{i}", "domain": domain, "split": split, "text": t, "aspects": []}
                for i, t in enumerate(picked)]
            stats["no_aspect"][f"{domain}_{split}"] = {"requested": n, "added": len(picked), "pool": len(pool)}
            assert len(picked) == n, f"aspect-free pool too small for {domain} {split}"

    texts = {sp: {r["text"].lower() for r in rows} for sp, rows in out.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        print(f"cross-split duplicate sentences {a}/{b}:", len(texts[a] & texts[b]))
    stats["splits"] = {}
    for split, rows in out.items():
        print(split, "within-split duplicate sentences:", len(rows) - len(texts[split]))
        write_jsonl(rows, PROCESSED / f"{split}.jsonl")
        c = Counter(a["sentiment"] for r in rows for a in r["aspects"])
        n_free = sum(1 for r in rows if not r["aspects"])
        print(split, "sentences", len(rows), "aspects", sum(c.values()), dict(c), "no-aspect", n_free)
        stats["splits"][split] = {"sentences": len(rows), "aspects": sum(c.values()), "sentiments": dict(c),
                                  "no_aspect_sentences": n_free,
                                  "no_aspect_share": round(n_free / len(rows), 4)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "source_preprocessing_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
