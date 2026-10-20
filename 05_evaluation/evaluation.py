# ============================================================
# Evaluation Script — Zero-Shot LLM Classification
# Iran Crisis TikTok Dataset
# Compares model outputs against gold standard (400 rows)
# Models: Mistral 7B, Qwen2.5-3B, DeepSeek-R1, Twitter-RoBERTa
# Note: DeepSeek classification is partial; its scores use only the gold rows it covers
# ============================================================

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support
)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

GOLD_FILE     = "data/clean/gold_standard_dataset.csv"
MISTRAL_FILE  = "data/clean/iran_crisis_videos_classified_mistral.csv"
QWEN_FILE     = "data/clean/iran_crisis_videos_classified_qwen.csv"
ROBERTA_FILE  = "data/clean/iran_crisis_videos_classified_roberta.csv"
DEEPSEEK_FILE = "data/clean/iran_crisis_videos_classified_deepseek.csv"
OUTPUT_FILE   = "data/clean/evaluation_results.txt"

# ── TEE ───────────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

log_file   = open(OUTPUT_FILE, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def header(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")

def subheader(text):
    print(f"\n--- {text} ---")

def norm_id(s):
    import re
    # Extract first 18 digits to handle scientific notation and trailing zeros
    return re.sub(r'[^0-9]', '', str(s).strip())[:18]

def norm_label(s):
    s = str(s).strip()
    s = s.replace('Humor', 'Humour')
    return s

def interpret_kappa(k):
    if k >= 0.81:   return "Almost perfect"
    elif k >= 0.61: return "Substantial"
    elif k >= 0.41: return "Moderate"
    elif k >= 0.21: return "Fair"
    else:           return "Slight/Poor"

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

print("Loading files...")
gold     = pd.read_csv(GOLD_FILE,     low_memory=False)
mistral  = pd.read_csv(MISTRAL_FILE,  low_memory=False)
qwen     = pd.read_csv(QWEN_FILE,     low_memory=False)
roberta  = pd.read_csv(ROBERTA_FILE,  low_memory=False)
deepseek = pd.read_csv(DEEPSEEK_FILE, low_memory=False)

for df in [gold, mistral, qwen, roberta, deepseek]:
    df['id_norm'] = df['id'].apply(norm_id)

print(f"Gold standard : {len(gold)} rows")
print(f"Mistral       : {len(mistral)} rows")
print(f"Qwen          : {len(qwen)} rows")
print(f"RoBERTa       : {len(roberta)} rows")
print(f"DeepSeek      : {len(deepseek)} rows (classification in progress)")

# ── MERGE ─────────────────────────────────────────────────────────────────────

df_mistral  = gold.merge(mistral[['id_norm','sentiment_label','frame_label']],  on='id_norm', how='inner')
df_qwen     = gold.merge(qwen[['id_norm','sentiment_label','frame_label']],     on='id_norm', how='inner')
df_roberta  = gold.merge(roberta[['id_norm','roberta_label','roberta_score']], on='id_norm', how='inner')
df_deepseek = gold.merge(deepseek[['id_norm','sentiment_label','frame_label']], on='id_norm', how='inner')

print(f"\nRows matched with gold standard:")
print(f"  Mistral  : {len(df_mistral)} / {len(gold)} ({len(df_mistral)/len(gold)*100:.1f}%)")
print(f"  Qwen     : {len(df_qwen)} / {len(gold)} ({len(df_qwen)/len(gold)*100:.1f}%)")
print(f"  RoBERTa  : {len(df_roberta)} / {len(gold)} ({len(df_roberta)/len(gold)*100:.1f}%)")
print(f"  DeepSeek : {len(df_deepseek)} / {len(gold)} ({len(df_deepseek)/len(gold)*100:.1f}%) ⚠ partial")

# ── NORMALIZE LABELS ──────────────────────────────────────────────────────────

for df in [df_mistral, df_qwen, df_deepseek]:
    df['gold_t1'] = df['tier1_sentiment'].apply(norm_label)
    df['gold_t3'] = df['tier3_frame'].apply(norm_label)
    df['pred_t1'] = df['sentiment_label'].apply(norm_label)
    df['pred_t3'] = df['frame_label'].apply(norm_label)

df_roberta['gold_t1'] = df_roberta['tier1_sentiment'].apply(norm_label)

roberta_map = {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}
df_roberta['pred_t1_roberta'] = df_roberta['roberta_label'].str.lower().map(roberta_map).fillna('Neutral')

def map_gold_to_3class(label):
    positive = {'Humour', 'Hope', 'Solidarity'}
    negative = {'Anger', 'Fear', 'Sadness', 'Disgust', 'Frustration', 'Curiosity'}
    if label in positive: return 'Positive'
    if label in negative: return 'Negative'
    return 'Neutral'

df_roberta['gold_3class'] = df_roberta['gold_t1'].apply(map_gold_to_3class)

# ── EVALUATE FUNCTION ─────────────────────────────────────────────────────────

def evaluate(gold_labels, pred_labels, model_name, tier_name, labels=None, note=""):
    gold_arr = gold_labels.values
    pred_arr = pred_labels.values
    if labels is None:
        labels = sorted(set(gold_arr) | set(pred_arr))

    try:
        kappa = cohen_kappa_score(gold_arr, pred_arr)
    except Exception:
        kappa = float('nan')

    agree = (gold_arr == pred_arr).sum()
    total = len(gold_arr)

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        gold_arr, pred_arr, average='weighted', labels=labels, zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        gold_arr, pred_arr, average='macro', labels=labels, zero_division=0)

    subheader(f"{model_name} — {tier_name}{note}")
    print(f"  Cohen's Kappa      : {kappa:.4f}  ({interpret_kappa(kappa)})")
    print(f"  Raw agreement      : {agree}/{total} ({agree/total*100:.1f}%)")
    print(f"  Weighted F1        : {f1_w:.4f}")
    print(f"  Weighted Precision : {prec_w:.4f}")
    print(f"  Weighted Recall    : {rec_w:.4f}")
    print(f"  Macro F1           : {f1_m:.4f}")

    print(f"\n  Per-label breakdown:")
    print(f"  {'Label':<25} {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Gold':>6}  {'Pred':>6}")
    print(f"  {'-' * 68}")

    prec_per, rec_per, f1_per, support = precision_recall_fscore_support(
        gold_arr, pred_arr, labels=labels, zero_division=0)
    pred_counts = pd.Series(pred_arr).value_counts()

    for i, label in enumerate(labels):
        g_cnt = support[i]
        p_cnt = pred_counts.get(label, 0)
        if g_cnt == 0 and p_cnt == 0:
            continue
        print(f"  {label:<25} {prec_per[i]:>10.3f}  {rec_per[i]:>8.3f}  {f1_per[i]:>8.3f}  {g_cnt:>6}  {p_cnt:>6}")

    return kappa, f1_w, prec_w, rec_w, agree, total

def print_confusion(gold_labels, pred_labels, labels, title):
    print(f"\n  Confusion Matrix — {title}")
    print(f"  (Rows = Gold standard, Columns = Model prediction)")
    cm    = confusion_matrix(gold_labels, pred_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "Gold \\ Pred"
    short = {l: l[:12] for l in labels}
    cm_df = cm_df.rename(index=short, columns=short)
    print(cm_df.to_string(max_cols=15))

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

header("TIER 1 — SENTIMENT EVALUATION")

t1_labels = sorted(df_mistral['gold_t1'].unique())
print(f"\nGold standard labels ({len(t1_labels)}): {t1_labels}")
print(f"Note: Neutral dominates at 94.2% (377/400 rows)")

results_t1 = {}
results_t1['Mistral']  = evaluate(df_mistral['gold_t1'],  df_mistral['pred_t1'],
                                   'Mistral 7B',  'Tier 1 Sentiment', t1_labels)
results_t1['Qwen']     = evaluate(df_qwen['gold_t1'],     df_qwen['pred_t1'],
                                   'Qwen2.5-3B',  'Tier 1 Sentiment', t1_labels)
results_t1['DeepSeek'] = evaluate(df_deepseek['gold_t1'], df_deepseek['pred_t1'],
                                   'DeepSeek-R1', 'Tier 1 Sentiment', t1_labels,
                                   note=f" [partial: {len(df_deepseek)}/{len(gold)} gold rows]")

print_confusion(df_mistral['gold_t1'],  df_mistral['pred_t1'],  t1_labels, 'Mistral 7B — Tier 1')
print_confusion(df_qwen['gold_t1'],     df_qwen['pred_t1'],     t1_labels, 'Qwen2.5-3B — Tier 1')
print_confusion(df_deepseek['gold_t1'], df_deepseek['pred_t1'], t1_labels, 'DeepSeek-R1 — Tier 1 (partial)')

# ─────────────────────────────────────────────────────────────────────────────
# ROBERTA — SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

header("TWITTER-ROBERTA — SENTIMENT EVALUATION (3-class baseline)")
print("\nNote: RoBERTa uses 3 classes (Positive/Negative/Neutral).")
print("Gold standard 10-class labels mapped to 3-class for comparison:")
print("  Positive → Humour, Hope, Solidarity")
print("  Negative → Anger, Fear, Sadness, Disgust, Frustration, Curiosity")
print("  Neutral  → Neutral")

agree_rob = (df_roberta['gold_3class'] == df_roberta['pred_t1_roberta']).sum()
total_rob  = len(df_roberta)

prec_rob, rec_rob, f1_rob, _ = precision_recall_fscore_support(
    df_roberta['gold_3class'], df_roberta['pred_t1_roberta'],
    average='weighted', labels=['Positive','Negative','Neutral'], zero_division=0)
try:
    kappa_rob = cohen_kappa_score(df_roberta['gold_3class'], df_roberta['pred_t1_roberta'])
except Exception:
    kappa_rob = float('nan')

print(f"\n  Cohen's Kappa      : {kappa_rob:.4f}  ({interpret_kappa(kappa_rob)})")
print(f"  Raw agreement      : {agree_rob}/{total_rob} ({agree_rob/total_rob*100:.1f}%)")
print(f"  Weighted F1        : {f1_rob:.4f}")
print(f"  Weighted Precision : {prec_rob:.4f}")
print(f"  Weighted Recall    : {rec_rob:.4f}")
print(f"  Mean confidence    : {df_roberta['roberta_score'].mean():.3f}")

print_confusion(df_roberta['gold_3class'], df_roberta['pred_t1_roberta'],
                ['Positive','Negative','Neutral'], 'RoBERTa — Tier 1 (3-class)')

# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — FRAME
# ─────────────────────────────────────────────────────────────────────────────

header("TIER 3 — NARRATIVE FRAME EVALUATION")

t3_labels = sorted(df_mistral['gold_t3'].unique())
print(f"\nGold standard labels ({len(t3_labels)}): {t3_labels}")
print(f"Note: RoBERTa does not classify narrative frame — excluded from Tier 3 evaluation")

results_t3 = {}
results_t3['Mistral']  = evaluate(df_mistral['gold_t3'],  df_mistral['pred_t3'],
                                   'Mistral 7B',  'Tier 3 Frame', t3_labels)
results_t3['Qwen']     = evaluate(df_qwen['gold_t3'],     df_qwen['pred_t3'],
                                   'Qwen2.5-3B',  'Tier 3 Frame', t3_labels)
results_t3['DeepSeek'] = evaluate(df_deepseek['gold_t3'], df_deepseek['pred_t3'],
                                   'DeepSeek-R1', 'Tier 3 Frame', t3_labels,
                                   note=f" [partial: {len(df_deepseek)}/{len(gold)} gold rows]")

print_confusion(df_mistral['gold_t3'],  df_mistral['pred_t3'],  t3_labels, 'Mistral 7B — Tier 3')
print_confusion(df_qwen['gold_t3'],     df_qwen['pred_t3'],     t3_labels, 'Qwen2.5-3B — Tier 3')
print_confusion(df_deepseek['gold_t3'], df_deepseek['pred_t3'], t3_labels, 'DeepSeek-R1 — Tier 3 (partial)')

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

header("SUMMARY — ALL MODELS")

print(f"\n{'Model':<15} {'Tier':<8} {'N':>5}  {'Kappa':>8}  {'Interp':<16}  {'Wtd F1':>8}  {'Agree%':>8}")
print("-" * 80)

for model, df in [('Mistral', df_mistral), ('Qwen', df_qwen), ('DeepSeek', df_deepseek)]:
    k, f1, prec, rec, agree, total = results_t1[model]
    flag = " ⚠" if model == 'DeepSeek' else ""
    print(f"  {model:<13} {'Tier 1':<8} {total:>5}  {k:>8.4f}  {interpret_kappa(k):<16}  {f1:>8.4f}  {agree/total*100:>7.1f}%{flag}")

print(f"  {'RoBERTa':<13} {'Tier 1':<8} {total_rob:>5}  {kappa_rob:>8.4f}  {interpret_kappa(kappa_rob):<16}  {f1_rob:>8.4f}  {agree_rob/total_rob*100:>7.1f}%")

print()
for model, df in [('Mistral', df_mistral), ('Qwen', df_qwen), ('DeepSeek', df_deepseek)]:
    k, f1, prec, rec, agree, total = results_t3[model]
    flag = " ⚠" if model == 'DeepSeek' else ""
    print(f"  {model:<13} {'Tier 3':<8} {total:>5}  {k:>8.4f}  {interpret_kappa(k):<16}  {f1:>8.4f}  {agree/total*100:>7.1f}%{flag}")

print(f"\n⚠ DeepSeek evaluated on {len(df_deepseek)}/{len(gold)} gold rows only (28.5%) — results not directly comparable to Mistral/Qwen")

# ── CLOSE ─────────────────────────────────────────────────────────────────────

sys.stdout = sys.__stdout__
log_file.close()
print(f"\nFull results saved to: {OUTPUT_FILE}")
print("--- DONE ---")
