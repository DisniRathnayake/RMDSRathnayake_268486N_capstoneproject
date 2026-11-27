"""Invariants of the processed source data (run after `python -m src.prepare_data`)."""
import re
from src.common import PROCESSED, read_jsonl
from src.metrics import norm
from src.textnorm import detokenize, detokenize_text, ctx_tokens


def _rows(split):
    return read_jsonl(PROCESSED / f"{split}.jsonl")


def test_detokenize_basic():
    assert detokenize_text("Boot time is super fast , around 35 seconds .") == "Boot time is super fast, around 35 seconds."
    assert detokenize_text("I do n't like it , but it ca n't be helped .") == "I don't like it, but it can't be helped."
    assert detokenize_text("Mother 's Day and the maitre d ' was rude") == "Mother's Day and the maitre d' was rude"
    assert detokenize_text("a `` junk '' app ( really ) for $ 150 .") == 'a "junk" app (really) for $150.'


def test_detokenize_spans_are_exact():
    toks = "The battery life is great , and it does n't overheat .".split(" ")
    text, spans = detokenize(toks)
    assert text[spans[1][0]:spans[2][1]] == "battery life"
    assert text[spans[9][0]:spans[10][1]] == "n't overheat"          # tokens 9..10 = "n't overheat"


def test_ctx_tokens_find_aspect_in_natural_text():
    toks, a = ctx_tokens("Great battery life."), ctx_tokens("life")
    assert any(toks[i:i + len(a)] == a for i in range(len(toks)))


def test_no_tokenisation_artefacts_left():
    for sp in ("train", "dev", "test"):
        rows = _rows(sp)
        bad = [r["text"] for r in rows if re.search(r" n't|\s[.,!?]$|``", r["text"])]
        assert len(bad) <= 0.005 * len(rows), (sp, bad[:3])      # a handful of untidy source sentences is fine


def test_aspects_and_reasons_are_spans_of_the_sentence():
    for sp in ("train", "dev", "test"):
        for r in _rows(sp):
            low = r["text"].lower()
            for a in r["aspects"]:
                assert a["aspect"].lower() in low and a["reason"].lower() in low
                assert a["reason"] in a["reasons"]


def test_no_duplicate_aspect_sentiment_entries():
    for sp in ("train", "dev", "test"):
        for r in _rows(sp):
            keys = [(norm(a["aspect"]), a["sentiment"]) for a in r["aspects"]]
            assert len(keys) == len(set(keys)), r["id"]


def test_dev_and_test_contain_aspect_free_sentences():
    shares = {}
    for sp in ("train", "dev", "test"):
        rows = _rows(sp)
        shares[sp] = sum(1 for r in rows if not r["aspects"]) / len(rows)
    assert shares["dev"] > 0.1 and shares["test"] > 0.1
    assert abs(shares["dev"] - shares["train"]) < 0.03 and abs(shares["test"] - shares["train"]) < 0.03


def test_splits_are_disjoint():
    texts = {sp: {r["text"].lower() for r in _rows(sp)} for sp in ("train", "dev", "test")}
    assert not texts["train"] & texts["dev"]
    assert not texts["train"] & texts["test"]
    assert not texts["dev"] & texts["test"]
