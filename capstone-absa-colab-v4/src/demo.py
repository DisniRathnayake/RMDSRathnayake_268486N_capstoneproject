"""Live demo for the video: run the extractors on a few reviews and show the JSON they produce.

Run:  python -m src.demo                                 (baseline only, no GPU)
      python -m src.demo --methods baseline prompted finetuned   (GPU; loads the model once per method)
"""
import argparse, json
from .extractors import get_extractor, extract_reviews, spacy_splitter

EXAMPLES = [
    "The fabric feels cheap and the sleeves are too short, but the color is beautiful.",
    "Runs small. I ordered my usual size and it was tight across the shoulders. Great customer service though, they exchanged it quickly.",
    "These boots are comfortable and the leather looks great. Sole wore out after two months.",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["baseline"])
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--adapter", default="adapters/absa-qlora")
    a = ap.parse_args()
    split = spacy_splitter()
    for m in a.methods:
        print(f"\n=== {m} ===")
        res = extract_reviews(EXAMPLES, get_extractor(m, a.model, a.adapter), split)
        for text, r in zip(EXAMPLES, res):
            print("\nREVIEW:", text)
            print("VALID JSON:", r["ok"], "| OUTPUT:", json.dumps(r["items"], indent=1))
