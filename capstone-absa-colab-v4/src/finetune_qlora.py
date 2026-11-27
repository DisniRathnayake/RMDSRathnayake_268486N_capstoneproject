"""QLoRA (4-bit) fine-tuning of an open-source LLM on the source-domain instruction pairs.

NOT run in the authoring environment (needs a GPU). Fallback for small GPUs: --model meta-llama/Llama-3.2-3B-Instruct
Run:  python -m src.build_sft_data && python -m src.finetune_qlora --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse, glob, torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)
from .common import PROCESSED, SEED, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
ap.add_argument("--out", default="adapters/absa-qlora")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--max_len", type=int, default=512)
args = ap.parse_args()
set_seed()
DTYPE = torch.bfloat16 if (torch.cuda.get_device_capability()[0] >= 8) else torch.float16   # T4 (Colab free) has no bf16
print("compute dtype:", DTYPE)

tok = AutoTokenizer.from_pretrained(args.model)
tok.pad_token = tok.pad_token or tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()


def encode(ex):
    p = tok(ex["prompt"], add_special_tokens=True)["input_ids"]
    c = tok(ex["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + c)[: args.max_len]
    labels = ([-100] * len(p) + c)[: args.max_len]      # loss on the completion only
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


ds = load_dataset("json", data_files={"train": str(PROCESSED / "sft_train.jsonl"),
                                      "dev": str(PROCESSED / "sft_dev.jsonl")})
ds = ds.map(encode, remove_columns=ds["train"].column_names)
Trainer(model=model,
        args=TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
                               per_device_train_batch_size=args.batch, gradient_accumulation_steps=4,
                               eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2, logging_steps=20, bf16=(DTYPE == torch.bfloat16), fp16=(DTYPE == torch.float16),
                               seed=SEED, report_to="none", lr_scheduler_type="cosine", warmup_steps=50),
        train_dataset=ds["train"], eval_dataset=ds["dev"],
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train(resume_from_checkpoint=bool(glob.glob(args.out + "/checkpoint-*")))
model.save_pretrained(args.out)
tok.save_pretrained(args.out)
