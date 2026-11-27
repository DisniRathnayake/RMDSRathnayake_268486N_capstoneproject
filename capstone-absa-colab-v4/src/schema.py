"""Output schema, prompt and validator shared by the prompted and fine-tuned LLM extractors."""
import json, re
from .common import SENTIMENTS

SYSTEM = ("You extract product aspects from a review. Return ONLY a JSON list. Each item has "
          '"aspect" (a phrase copied from the review), "sentiment" (positive, negative or neutral) '
          'and "reason" (the opinion words copied from the review). Return [] if there is no aspect.')

FEW_SHOT = [
    ("The battery life is great but the screen is too dim.",
     [{"aspect": "battery life", "sentiment": "positive", "reason": "great"},
      {"aspect": "screen", "sentiment": "negative", "reason": "too dim"}]),
    ("Service was slow and the staff were rude.",
     [{"aspect": "Service", "sentiment": "negative", "reason": "slow"},
      {"aspect": "staff", "sentiment": "negative", "reason": "rude"}]),
    ("I bought this last week.", []),
]


def build_prompt(review: str, few_shot=True) -> str:
    parts = [SYSTEM, ""]
    if few_shot:
        for r, out in FEW_SHOT:
            parts += [f"Review: {r}", f"Output: {json.dumps(out)}", ""]
    parts += [f"Review: {review}", "Output:"]
    return "\n".join(parts)


def parse_output(raw: str):
    """Extract the first JSON list in the text. Returns the list or None."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\[", raw):               # first '[' that starts a parseable JSON list
        try:
            data, _ = dec.raw_decode(raw[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data
    return None


def validate(items, review: str):
    """Schema + span check. Returns (ok, error)."""
    if items is None:
        return False, "not a JSON list"
    low = review.lower()
    for it in items:
        if not isinstance(it, dict) or not {"aspect", "sentiment", "reason"} <= set(it):
            return False, "missing keys"
        if it["sentiment"] not in SENTIMENTS:
            return False, "bad sentiment"
        if not isinstance(it["aspect"], str) or it["aspect"].lower() not in low:
            return False, "aspect not a span of the review"
        if not isinstance(it["reason"], str) or it["reason"].lower() not in low:
            return False, "reason not a span of the review"
    return True, ""
