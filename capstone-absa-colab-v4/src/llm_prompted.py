"""Few-shot prompted open-source LLM extractor (local, 4-bit). No external APIs.

`extract_with_retry` works with any generate(prompt)->str function, so the same code evaluates the
prompted model and the QLoRA fine-tuned model (pass fine_tuned=True and the adapter path).
Invalid output gets one retry, then is logged as a failure and excluded from that method's score.

Run (GPU):  python -m src.llm_prompted --split dev --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse, json, time
from .common import PROCESSED, RESULTS, read_jsonl, set_seed
from .schema import build_prompt, parse_output, validate
from .metrics import evaluate, bootstrap_ci


def _gen(generate, prompt, sample):
    if not sample:
        return generate(prompt)
    try:
        return generate(prompt, sample=True)     # retry with light sampling: greedy would repeat the same failure
    except TypeError:
        return generate(prompt)                  # mock generators without the kwarg


def extract_with_retry(review, generate, few_shot=True):
    raw = ""
    for attempt in range(2):                     # first try + one retry
        raw = _gen(generate, build_prompt(review, few_shot), sample=attempt > 0)
        items = parse_output(raw)
        ok, err = validate(items, review)
        if ok:
            return {"ok": True, "items": items, "raw": raw, "attempts": attempt + 1}
    return {"ok": False, "items": [], "raw": raw, "attempts": 2, "error": err}


def hf_generator(model_name, adapter=None, max_new_tokens=200):
    """Local 4-bit generator via transformers + bitsandbytes (lazy imports, needs a GPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_name)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16 if (torch.cuda.get_device_capability()[0] >= 8) else torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def generate(prompt, sample=False):
        ids = tok(prompt, return_tensors="pt").to(model.device)
        kw = dict(do_sample=True, temperature=0.3, top_p=0.9) if sample else dict(do_sample=False)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
                                 stop_strings=["\nReview:", "\n\n"], tokenizer=tok, **kw)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.split("\nReview:")[0]          # drop any continued few-shot blocks
    return generate


def run(rows, generate, few_shot=True):
    results, t0 = [], time.time()
    for r in rows:
        results.append(extract_with_retry(r["text"], generate, few_shot))
    secs = (time.time() - t0) / max(len(rows), 1)
    return results, secs


def score(rows, results, seconds_per_review, name):
    valid = [(r, x) for r, x in zip(rows, results) if x["ok"]]
    preds = [x["items"] for _, x in valid]
    golds = [r["aspects"] for r, _ in valid]
    out = {"method": name, "n": len(rows), "json_validity_rate": len(valid) / max(len(rows), 1),
           "seconds_per_review": seconds_per_review, "metrics": evaluate(preds, golds),
           # strict: every sentence counts, an invalid output is an empty prediction (no survivorship bias)
           "metrics_strict_all": evaluate([x["items"] for x in results], [r["aspects"] for r in rows]),
           "metrics_scope_note": "metrics = valid outputs only; metrics_strict_all = all sentences"}
    if preds:
        out["pair_exact_f1_ci"] = bootstrap_ci(preds, golds, n_boot=300)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (fine-tuned run)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed()
    rows = read_jsonl(PROCESSED / f"{args.split}.jsonl")[: args.limit]
    gen = hf_generator(args.model, args.adapter)
    name = "finetuned" if args.adapter else "prompted"
    results, secs = run(rows, gen, few_shot=not args.adapter)
    out = score(rows, results, secs, name)
    RESULTS.mkdir(exist_ok=True)
    from .common import write_jsonl
    write_jsonl([{"id": r["id"], "text": r["text"], "gold": r["aspects"], "ok": x["ok"], "items": x["items"],
                  "raw": x["raw"], "error": x.get("error", "")} for r, x in zip(rows, results)],
                RESULTS / f"preds_{name}_{args.split}.jsonl")
    (RESULTS / f"{name}_{args.split}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "metrics"}, indent=2))
    print(json.dumps(out["metrics"]["pair_exact"], indent=2))
