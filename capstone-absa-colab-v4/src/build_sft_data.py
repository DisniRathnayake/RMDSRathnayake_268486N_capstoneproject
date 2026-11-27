"""Convert the unified train/dev data into instruction pairs for QLoRA fine-tuning.

Run:  python -m src.build_sft_data
"""
import json
from .common import PROCESSED, read_jsonl, write_jsonl
from .schema import build_prompt


def to_pair(r):
    target = [{"aspect": a["aspect"], "sentiment": a["sentiment"], "reason": a["reason"]} for a in r["aspects"]]
    return {"id": r["id"], "prompt": build_prompt(r["text"], few_shot=False), "completion": " " + json.dumps(target)}


if __name__ == "__main__":
    for split in ("train", "dev"):
        pairs = [to_pair(r) for r in read_jsonl(PROCESSED / f"{split}.jsonl")]
        write_jsonl(pairs, PROCESSED / f"sft_{split}.jsonl")
        print(split, len(pairs))
