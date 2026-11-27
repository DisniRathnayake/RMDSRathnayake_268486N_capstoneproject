"""Shared helpers: seeds, paths, unified record schema and JSONL IO.

Unified record:
  {"id": str, "domain": "laptops"|"restaurants"|"clothing", "split": str, "text": str,
   "aspects": [{"aspect": str, "sentiment": "positive"|"negative"|"neutral", "reason": str}]}
"""
import json, os, random
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
SENTIMENTS = ("positive", "negative", "neutral")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
