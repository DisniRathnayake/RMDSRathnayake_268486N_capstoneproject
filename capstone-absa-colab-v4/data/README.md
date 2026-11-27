# Data layout
- `external/aste_v2/`: ASTE-Data-V2 triplet files (Xu et al., 2020), derived from SemEval-2014 Task 4. Public benchmark data, cite the original papers.
- `external/semeval2014/`: SemEval-2014 train XML (used only for no-aspect sentences).
- `processed/`: unified train/dev/test JSONL (`python -m src.prepare_data`) and instruction pairs (`python -m src.build_sft_data`).
- `raw/`: clothing review text. Git-ignored; created by `scripts/fetch_data.py`.
- `annotation/`: annotation sheets (git-ignored) and `gold_labels.jsonl` (IDs and labels only, publishable).
