# Twitter-RoBERTa Classification Script
# Classifies video descriptions for Tier 1 Sentiment (Baseline)
# RoBERTa only provides 3 labels: positive, negative, neutral

import os
import pandas as pd
from transformers import pipeline

# ── CONFIGURATION ───

INPUT_FILE  = "data/clean/iran_crisis_videos_all_clean.csv"
OUTPUT_FILE = "data/clean/iran_crisis_videos_classified_roberta.csv"

# Save progress every N rows
SAVE_EVERY  = 100

# Model
MODEL_NAME  = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# ── LOAD MODEL ───

print("---Loading Twitter-RoBERTa model---")
print("(Model downloads on first run — cached after that)")

classifier = pipeline(
    "sentiment-analysis",
    model     = MODEL_NAME,
    truncation = True,
    max_length = 512
)

print("Model loaded successfully.")

# ── LOAD DATA ───

print("---Loading data---")
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
print(f"Loaded {len(df)} rows")

# Check if partial output exists — resume from where we left off
if os.path.exists(OUTPUT_FILE):
    df_existing = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
    start_idx   = len(df_existing)
    print(f"Resuming from row {start_idx} — {len(df) - start_idx} rows remaining")
    df_remaining = df.iloc[start_idx:].reset_index(drop=True)
    results      = df_existing.to_dict("records")
else:
    start_idx    = 0
    df_remaining = df.copy()
    results      = []
    print("Starting fresh classification")

# ── CLASSIFICATION LOOP ───

print(f"---Starting classification on {len(df_remaining)} rows---")
total = len(df_remaining)

for i, row in df_remaining.iterrows():
    text = str(row.get("video_description_english", ""))

    try:
        if len(text.strip()) < 3:
            roberta_label = "neutral"
            roberta_score = 0.0
        else:
            result        = classifier(text[:512])[0]
            roberta_label = result["label"].lower()
            roberta_score = round(result["score"], 4)
    except Exception as e:
        print(f"  Error on row {i}: {e}")
        roberta_label = "neutral"
        roberta_score = 0.0

    result_row                  = row.to_dict()
    result_row["roberta_label"] = roberta_label
    result_row["roberta_score"] = roberta_score
    result_row["model_used"]    = MODEL_NAME

    results.append(result_row)

    # Progress update every 50 rows
    if (i + 1) % 50 == 0 or (i + 1) == total:
        print(f"  Processed {i + 1}/{total} rows — last: [{roberta_label}] (score: {roberta_score})")

    # Save progress every SAVE_EVERY rows
    if (i + 1) % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"  Progress saved at row {start_idx + i + 1}")

# ── SAVE FINAL OUTPUT ──

df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n---Classification complete---")
print(f"Output saved to: {OUTPUT_FILE}")

# ── SUMMARY ──

print("\n---CLASSIFICATION SUMMARY---")
print(f"Model used      : {MODEL_NAME}")
print(f"Total rows      : {len(df_results)}")
print(f"\nSentiment distribution (3-class baseline):")
print(df_results["roberta_label"].value_counts().to_string())
print(f"\nAverage confidence score: {df_results['roberta_score'].mean():.4f}")
print(f"\nNote: RoBERTa provides positive/negative/neutral only.")
print(f"Use Mistral and Qwen results for full 10-label sentiment classification.")