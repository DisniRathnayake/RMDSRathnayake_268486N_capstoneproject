"""
Fetch the datasets named in the Milestone 1 proposal. Run on a machine with normal internet access.

  python fetch_proposal_data.py amazon  --n 5000 --scan 400000 --out amazon_clothing_sample.jsonl
  python fetch_proposal_data.py semeval-gold --out semeval_gold

amazon: streams the Amazon Reviews 2023 Clothing_Shoes_and_Jewelry file from Hugging Face (about 9 GB in
        total, but only the first --scan lines are read), reservoir-samples --n reviews, drops reviewer
        identifiers (compliance point in the proposal) and keeps a hashed review_id.
        Use --local FILE to run the same filtering on a local .jsonl instead.
semeval-gold: downloads the official gold test XML files (with polarity labels).
"""
import argparse, hashlib, json, os, random, re, sys, urllib.request

AMAZON_URL = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
              "raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl")
GOLD_BASE = "https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/resolve/main/ABSA_Gold_TestData/"
GOLD_FILES = ["Laptops_Test_Gold.xml", "Restaurants_Test_Gold.xml"]

def lines_from_url(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "capstone-fetch"})) as r:
        for raw in r:
            yield raw.decode("utf-8", errors="replace")

def lines_from_file(path):
    with open(path, encoding="utf-8") as f:
        yield from f

def keep(r):
    text = (r.get("text") or "").strip()
    n = len(text.split())
    if not (15 <= n <= 300):
        return None
    if sum(c.isascii() for c in text) / max(len(text), 1) < 0.95:   # crude English filter
        return None
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    rid = hashlib.sha1(f'{r.get("parent_asin")}|{r.get("timestamp")}|{text[:40]}'.encode()).hexdigest()[:12]
    return {"review_id": "amz_" + rid, "parent_asin": r.get("parent_asin"), "rating": r.get("rating"),
            "title": r.get("title"), "text": text, "verified_purchase": r.get("verified_purchase")}

def amazon(args):
    rng = random.Random(args.seed)
    src = lines_from_file(args.local) if args.local else lines_from_url(AMAZON_URL)
    sample, seen, k = [], set(), 0
    for i, line in enumerate(src):
        if i >= args.scan:
            break
        try:
            r = keep(json.loads(line))
        except json.JSONDecodeError:
            continue
        if r is None or r["text"] in seen:
            continue
        seen.add(r["text"])
        k += 1
        if len(sample) < args.n:
            sample.append(r)
        else:
            j = rng.randint(0, k - 1)
            if j < args.n:
                sample[j] = r
    with open(args.out, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"scanned {min(i + 1, args.scan)} lines, {k} eligible, wrote {len(sample)} to {args.out}")

def semeval_gold(args):
    os.makedirs(args.out, exist_ok=True)
    for name in GOLD_FILES:
        dest = os.path.join(args.out, name)
        urllib.request.urlretrieve(GOLD_BASE + name, dest)
        print("saved", dest, os.path.getsize(dest), "bytes")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("amazon")
    a.add_argument("--n", type=int, default=5000)
    a.add_argument("--scan", type=int, default=400000)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--local")
    a.add_argument("--out", default="amazon_clothing_sample.jsonl")
    g = sub.add_parser("semeval-gold")
    g.add_argument("--out", default="semeval_gold")
    args = p.parse_args()
    amazon(args) if args.cmd == "amazon" else semeval_gold(args)
