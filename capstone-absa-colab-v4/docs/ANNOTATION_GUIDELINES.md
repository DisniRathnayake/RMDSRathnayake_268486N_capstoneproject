# Clothing gold-set annotation guidelines

Goal: label every product aspect that the reviewer comments on, with its sentiment, for the clothing
target-domain evaluation set. Written before annotating so that decisions stay consistent.

**Format** (column `annotation`): `aspect|sentiment; aspect|sentiment`. Write `none` if the review has no aspect.

1. **Aspect term**: the shortest noun phrase in the review that names the feature, copied exactly (e.g. `fabric`,
   `sleeves`, `customer service`). Do not include the opinion word (`soft fabric` becomes `fabric`).
2. **Sentiment** is positive, negative or neutral, judged for that aspect only, not the whole review.
   Neutral means the aspect is mentioned with no evaluation (`the dress has pockets`).
3. **Product names are not aspects** (`dress`, `shirt`, `jacket`) unless the review evaluates the item as a whole
   through a feature word such as `overall quality`.
4. **Fit and size words count as aspects**: `fit`, `size`, `length`; `runs small` annotates as `fit|negative`
   or `size|negative` depending on the review wording. Pick the noun the reviewer used; if none, use `fit`.
5. **Mixed sentiment on one aspect** in one review: label the aspect once with the dominant sentiment and add `neutral` only if both
   sides are equal.
6. **Repeated mentions** of the same aspect in a review count once.
7. **Implicit aspects and sarcasm** are not annotated (known limitation, stated in the report).
8. **Pronoun references** (`it is too tight`) are labelled with the aspect they refer to only if the noun appears in the review.
9. After finishing batch 1, wait at least one week, then annotate `batch2_recheck.csv` without looking at batch 1 labels.

## Note added after annotation (v4)
Guidelines 1 and 7 ask for aspects copied from the review and no implicit aspects. The completed gold set deviates:
only 52% of gold aspects occur in the review text (40% as whole words) and 67 of 300 reviews have none at all, because
most labels are aspect categories inferred from opinion words (`comfortable` becomes `comfort`, `can be dressed up`
becomes `style`). The gold labels were not changed. Instead, evaluation is done at category level: extracted aspects
and gold labels are both mapped to canonical categories (`configs/canonical_aspects.json`, `src/canonical.py`) and
merged to one label per category per review (guidelines 5 and 6). A span-level score on the literal part of the gold
is reported as a secondary, like-for-like metric for the cross-domain drop.
