# Runbook: remaining Milestone 2 steps (run on your GPU machine)

Run from the repo root. Everything below uses seed 42 (set in `src/common.py`).

## 0. Setup (once)
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt && python -m spacy download en_core_web_sm
# Llama 3 is gated: accept the licence on the model page on Hugging Face, then
huggingface-cli login
nvidia-smi        # note the GPU model and VRAM for the README "Hardware" line
```
A ready-made Colab notebook is `notebooks/colab_run.ipynb`.
Model choice by VRAM: 8B in 4-bit needs about 8 to 10 GB for inference and 14 to 18 GB for QLoRA training.
With less, use `meta-llama/Llama-3.2-3B-Instruct` (about half), as allowed by the proposal. Use the same model for all LLM steps.

## 1. Data (already in the repo, rerun if you change anything)
v4: `prepare_data` detokenises the ASTE text, merges duplicate (aspect, sentiment) entries and adds aspect-free sentences to dev/test. Any adapter trained on older processed data must be retrained.
```bash
python -m src.prepare_data && python -m src.build_sft_data
python scripts/fetch_data.py amazon --n 5000 --out data/raw/amazon_clothing_sample.jsonl
python -m src.prepare_clothing
python -m src.baseline_spacy --split dev && python -m src.baseline_spacy --split test
```

## 2. Smoke test the prompted LLM (5 minutes)
```bash
python -m src.llm_prompted --split dev --limit 30 --model meta-llama/Meta-Llama-3-8B-Instruct
```
Check `results/prompted_dev.json`: JSON validity should be well above 0.8. If it is low, look at `raw` outputs (add a print in `extract_with_retry`) and shorten or adjust the prompt in `src/schema.py`.

## 3. Prompted LLM on the full source dev and test sets
```bash
python -m src.llm_prompted --split dev  --model meta-llama/Meta-Llama-3-8B-Instruct
python -m src.llm_prompted --split test --model meta-llama/Meta-Llama-3-8B-Instruct
```
Writes `results/prompted_dev.json` and `results/prompted_test.json` (F1, JSON validity, seconds per review, bootstrap CI).

## 4. QLoRA fine-tuning (start this early; it is the longest step)
```bash
python -m src.finetune_qlora --model meta-llama/Meta-Llama-3-8B-Instruct --epochs 3
```
Output adapter: `adapters/absa-qlora` (Colab: `/content/drive/MyDrive/absa-qlora-v4`) (git-ignored; do not commit weights, publish the adapter on the Hugging Face Hub if the proposal's "trained LoRA adapter" output needs a link).
Out-of-memory: lower `--batch` to 2 or 1, lower `--max_len` to 384, or switch to the 3B model.
Watch eval loss per epoch: if it rises after epoch 1 or 2, retrain with fewer epochs.
```bash
python -m src.llm_prompted --split dev  --model meta-llama/Meta-Llama-3-8B-Instruct --adapter adapters/absa-qlora
python -m src.llm_prompted --split test --model meta-llama/Meta-Llama-3-8B-Instruct --adapter adapters/absa-qlora
```
Writes `results/finetuned_dev.json` and `results/finetuned_test.json`. The `--model` must match the one used for training.

## 5. Clothing gold annotation (can be done while training runs)
1. Open `annotation_batch1.csv` (copy it to `data/annotation/batch1.csv`). Follow `docs/ANNOTATION_GUIDELINES.md`.
2. Fill the `annotation` column: `fit|negative; color|positive`, or `none`. Do at least 100 reviews for Milestone 2, all 300 for the final.
3. Wait at least a week, then fill `batch2_recheck.csv` without looking at batch 1.
4. `python scripts/parse_annotations.py` writes `data/annotation/gold_labels.jsonl` (IDs and labels only, safe to commit) and `results/annotation_consistency.json`.

## 6. Evaluate on clothing and compute the cross-domain drop
```bash
python -m src.eval_clothing --method baseline
python -m src.eval_clothing --method baseline_lr
python -m src.eval_clothing --method prompted  --model meta-llama/Meta-Llama-3-8B-Instruct
python -m src.eval_clothing --method finetuned --model meta-llama/Meta-Llama-3-8B-Instruct --adapter adapters/absa-qlora
python -m src.compare            # results/comparison.md, paste into the report
```
The drop is the source test span F1 minus the clothing span F1 on literal gold, so step 3 and 4 must be done first. The primary clothing numbers are category-level (see README, Evaluation on clothing); `results/clothing_<method>.json` also holds the net-sentiment agreement with the gold labels.

## 7. Aggregation and report view
```bash
python -m src.aggregate --method baseline --min_mentions 10
python -m src.aggregate --method finetuned --limit 300 --min_mentions 5     # slower: LLM over every sentence
streamlit run app.py
```
Aggregate agreement with gold (correlation of system and gold net sentiment per category) is computed by `src.eval_clothing` on the gold reviews and stored in `results/clothing_<method>.json` under `agreement`.

## 8. Before submitting
- Fill the hardware line in the README (GPU, VRAM, CPU, RAM, training time).
- Copy the numbers from `results/comparison.md` and the `results/*.json` files into the README.
- Commit. Check `git status` shows no files from `data/raw/`, `data/annotation/batch*.csv` or `adapters/`.
