"""Text normalisation shared by data preparation and the baselines.

ASTE-Data-V2 stores sentences PTB-tokenised ("battery life .", "do n't", "`` junk ''"), while the clothing
reviews are natural text ("battery life.", "don't"). `detokenize` converts the tokenised form back to natural
text AND returns the character span of every token, so aspect/opinion spans (token indices in ASTE) are cut
from the detokenised string and stay exact substrings of it.
"""
import re

NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "%", "...", "..", "''", "'",
                   "n't", "'s", "'re", "'ve", "'m", "'d", "'ll"}
NO_SPACE_AFTER = {"(", "[", "{", "$", "``"}
QUOTE_MAP = {"``": '"', "''": '"'}                # PTB quotes -> straight quotes (also covers 13 '' inches)


def detokenize(tokens):
    """tokens -> (text, spans) where spans[i] = (start, end) of token i in text."""
    out, spans, pos, prev = [], [], 0, None
    for tok in tokens:
        if tok == "":                             # empty token (double space): zero-width, no effect on spacing
            spans.append((pos, pos))
            continue
        glue = prev is not None and (tok.lower() in NO_SPACE_BEFORE or prev in NO_SPACE_AFTER)
        if prev is not None and not glue:
            out.append(" ")
            pos += 1
        piece = QUOTE_MAP.get(tok, tok)
        spans.append((pos, pos + len(piece)))
        out.append(piece)
        pos += len(piece)
        prev = tok
    return "".join(out), spans


def detokenize_text(text: str) -> str:
    return detokenize(text.split(" "))[0]


_CTX_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def ctx_tokens(s: str):
    """Lower-cased word/punctuation tokens, so an aspect is found in natural text even when punctuation is attached to it."""
    return _CTX_TOKEN.findall(s.lower())
