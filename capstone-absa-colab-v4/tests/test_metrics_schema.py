from src.metrics import evaluate, relaxed_match
from src.schema import parse_output, validate


def test_perfect_and_empty():
    g = [[{"aspect": "battery life", "sentiment": "positive"}], []]
    assert evaluate(g, g)["pair_exact"]["f1"] == 1.0
    assert evaluate([[], []], g)["pair_exact"]["f1"] == 0.0


def test_relaxed_vs_sentiment():
    assert relaxed_match("battery", "battery life")
    p = [[{"aspect": "battery", "sentiment": "negative"}]]
    g = [[{"aspect": "battery life", "sentiment": "positive"}]]
    assert evaluate(p, g)["aspect_relaxed"]["f1"] == 1.0
    assert evaluate(p, g)["pair_relaxed"]["f1"] == 0.0


def test_validator_span_check():
    review = "The screen is too dim."
    ok = parse_output('[{"aspect": "screen", "sentiment": "negative", "reason": "too dim"}]')
    assert validate(ok, review)[0]
    bad = parse_output('[{"aspect": "battery", "sentiment": "negative", "reason": "too dim"}]')
    assert not validate(bad, review)[0]
    assert not validate(parse_output("no json here"), review)[0]


def test_parser_ignores_continued_blocks():
    raw = ' [{"aspect": "screen", "sentiment": "negative", "reason": "dim"}]\n\nReview: x\nOutput: [{"aspect": "food", "sentiment": "positive", "reason": "great"}]'
    items = parse_output(raw)
    assert items is not None and len(items) == 1 and items[0]["aspect"] == "screen"


def test_relaxed_is_token_level_not_substring():
    assert not relaxed_match("tea", "steak")
    assert not relaxed_match("art", "party")
    assert relaxed_match("battery", "battery life")
    assert relaxed_match("life", "battery life")
    assert relaxed_match("battery life", "life of battery")      # Jaccard 2/3
    assert not relaxed_match("screen", "screenshot")
    p = [[{"aspect": "tea", "sentiment": "positive"}]]
    g = [[{"aspect": "steak", "sentiment": "positive"}]]
    assert evaluate(p, g)["aspect_relaxed"]["f1"] == 0.0
