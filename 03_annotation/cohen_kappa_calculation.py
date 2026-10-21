# ============================================================
# Cohen's Kappa Inter-Rater Reliability Calculation
# Iran Crisis TikTok Annotation Project
# Annotator 1: Gourav Kumawat
# Annotator 2: Dr. Anna Carolina Haensch
# ============================================================

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# --- TEE — write to both terminal and file simultaneously ------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

# --- CONFIGURATION --------------------------------------------------------------

GOURAV_FILE = "data/clean/annotation_done_by_me - annotation_sample_400.csv"
CARO_FILE   = "data/clean/annotation_sample_400.csv"
OUTPUT_FILE = "data/clean/kappa_results_full.txt"

# Open output file and start tee
log_file   = open(OUTPUT_FILE, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log_file)

# --- LOAD DATA -----------------------------------------------------------------

print("Loading annotation files...")
gourav = pd.read_csv(GOURAV_FILE, low_memory=False)
caro   = pd.read_csv(CARO_FILE,   low_memory=False)

print(f"Gourav file : {len(gourav)} rows")
print(f"Caro file   : {len(caro)} rows")

mismatch = 0
for i in range(len(gourav)):
    if str(gourav['video_description_clean'].iloc[i])[:30] != str(caro['video_description_clean'].iloc[i])[:30]:
        mismatch += 1
print(f"Row alignment check: {mismatch} mismatches (0 = perfect)")

# --- NORMALIZE LABELS -----------------------------------------------------------

gourav['tier1'] = gourav['tier1_sentiment'].str.strip()
gourav['tier3'] = gourav['tier3_frame'].str.strip()
caro['tier1']   = caro['tier1_sentiment'].str.strip()
caro['tier3']   = caro['tier3_frame'].str.strip().str.replace(' / ', '/', regex=False)

# --- HELPER ---------------------------------------------------------------------

def header(text):
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")

def interpret_kappa(k):
    if k >= 0.81:   return "Almost perfect agreement"
    elif k >= 0.61: return "Substantial agreement"
    elif k >= 0.41: return "Moderate agreement"
    elif k >= 0.21: return "Fair agreement"
    else:           return "Slight or poor agreement"

# --- OVERALL KAPPA -----------------------------------------------------------

header("OVERALL COHEN'S KAPPA")

kappa_t1  = cohen_kappa_score(gourav['tier1'], caro['tier1'])
kappa_t3  = cohen_kappa_score(gourav['tier3'], caro['tier3'])
agree_t1  = (gourav['tier1'] == caro['tier1']).sum()
agree_t3  = (gourav['tier3'] == caro['tier3']).sum()
total     = len(gourav)

print(f"\nAnnotator 1 : Gourav Kumawat")
print(f"Annotator 2 : Dr. Anna-Carolina Haensch")
print(f"Sample size : {total} rows")

print(f"\nTier 1 — Sentiment")
print(f"  Cohen's Kappa  : {kappa_t1:.4f}")
print(f"  Raw agreement  : {agree_t1}/{total} ({agree_t1/total*100:.1f}%)")
print(f"  Threshold      : κ ≥ 0.61")
print(f"  Status         : {'✓ PASS' if kappa_t1 >= 0.61 else '✗ BELOW THRESHOLD'}")

print(f"\nTier 3 — Narrative Frame")
print(f"  Cohen's Kappa  : {kappa_t3:.4f}")
print(f"  Raw agreement  : {agree_t3}/{total} ({agree_t3/total*100:.1f}%)")
print(f"  Threshold      : κ ≥ 0.61")
print(f"  Status         : {'✓ PASS' if kappa_t3 >= 0.61 else '✗ BELOW THRESHOLD'}")

print(f"\n  Tier 1 interpretation : {interpret_kappa(kappa_t1)}")
print(f"  Tier 3 interpretation : {interpret_kappa(kappa_t3)}")

# --- PER-LABEL KAPPA — TIER 1 -------------------------------------------------------------

header("TIER 1 — PER-LABEL KAPPA BREAKDOWN")

labels_t1 = sorted(set(gourav['tier1'].unique()) | set(caro['tier1'].unique()))

print(f"\n{'Label':<20} {'κ':>7}  {'Gourav':>8}  {'Caro':>6}  {'Both':>5}  {'Interpretation'}")
print("-" * 72)
for label in labels_t1:
    g_bin    = (gourav['tier1'] == label).astype(int)
    c_bin    = (caro['tier1']   == label).astype(int)
    both     = ((gourav['tier1'] == label) & (caro['tier1'] == label)).sum()
    g_cnt    = g_bin.sum()
    c_cnt    = c_bin.sum()
    try:
        k = cohen_kappa_score(g_bin, c_bin)
        interp = interpret_kappa(k) if (g_cnt + c_cnt) > 2 else "Too few samples"
        print(f"  {label:<18} {k:>7.3f}  {g_cnt:>8}  {c_cnt:>6}  {both:>5}  {interp}")
    except Exception:
        print(f"  {label:<18} {'N/A':>7}  {g_cnt:>8}  {c_cnt:>6}  {both:>5}  Cannot compute")

# -- PER-LABEL KAPPA — TIER 3 ------------------------------------------------------------

header("TIER 3 — PER-LABEL KAPPA BREAKDOWN")

labels_t3 = sorted(set(gourav['tier3'].unique()) | set(caro['tier3'].unique()))

print(f"\n{'Label':<27} {'κ':>7}  {'Gourav':>8}  {'Caro':>6}  {'Both':>5}  {'Interpretation'}")
print("-" * 77)
for label in labels_t3:
    g_bin    = (gourav['tier3'] == label).astype(int)
    c_bin    = (caro['tier3']   == label).astype(int)
    both     = ((gourav['tier3'] == label) & (caro['tier3'] == label)).sum()
    g_cnt    = g_bin.sum()
    c_cnt    = c_bin.sum()
    try:
        k = cohen_kappa_score(g_bin, c_bin)
        interp = interpret_kappa(k) if (g_cnt + c_cnt) > 2 else "Too few samples"
        print(f"  {label:<25} {k:>7.3f}  {g_cnt:>8}  {c_cnt:>6}  {both:>5}  {interp}")
    except Exception:
        print(f"  {label:<25} {'N/A':>7}  {g_cnt:>8}  {c_cnt:>6}  {both:>5}  Cannot compute")

# --- LABEL DISTRIBUTION COMPARISON ---------------------------------------------------------

header("LABEL DISTRIBUTION COMPARISON")

print(f"\n--- Tier 1 Sentiment ---")
print(f"\n{'Label':<20} {'Gourav (count)':>15}  {'Gourav (%)':>10}  {'Caro (count)':>13}  {'Caro (%)':>9}")
print("-" * 75)
for label in sorted(set(gourav['tier1'].unique()) | set(caro['tier1'].unique())):
    gc = (gourav['tier1'] == label).sum()
    cc = (caro['tier1']   == label).sum()
    print(f"  {label:<18} {gc:>15}  {gc/total*100:>9.1f}%  {cc:>13}  {cc/total*100:>8.1f}%")

print(f"\n--- Tier 3 Frame ---")
print(f"\n{'Label':<27} {'Gourav (count)':>15}  {'Gourav (%)':>10}  {'Caro (count)':>13}  {'Caro (%)':>9}")
print("-" * 80)
for label in sorted(set(gourav['tier3'].unique()) | set(caro['tier3'].unique())):
    gc = (gourav['tier3'] == label).sum()
    cc = (caro['tier3']   == label).sum()
    print(f"  {label:<25} {gc:>15}  {gc/total*100:>9.1f}%  {cc:>13}  {cc/total*100:>8.1f}%")

# -- DISAGREEMENT ANALYSIS — TIER 1 --------------------------------------------------------

header("TIER 1 — DISAGREEMENT ANALYSIS")

disagree_mask_t1 = gourav['tier1'] != caro['tier1']
disagree_t1 = gourav[disagree_mask_t1].copy()
disagree_t1['caro_label']    = caro.loc[disagree_mask_t1, 'tier1'].values
disagree_t1['text_preview']  = disagree_t1['video_description_english'].astype(str).str[:80]

print(f"\nTotal disagreements: {disagree_mask_t1.sum()} / {total} ({disagree_mask_t1.sum()/total*100:.1f}%)")

pairs = disagree_t1.groupby(['tier1', 'caro_label']).size().reset_index(name='count').sort_values('count', ascending=False)
print(f"\nMost common disagreement patterns:")
for _, row in pairs.iterrows():
    print(f"  Gourav: {row['tier1']:<15} → Caro: {row['caro_label']:<15} ({row['count']} times)")

print(f"\nAll Tier 1 disagreements:")
print(f"{'#':<4} {'Gourav':<15} {'Caro':<15} {'Text preview'}")
print("-" * 80)
for idx, (_, row) in enumerate(disagree_t1.iterrows(), 1):
    print(f"  {idx:<3} {row['tier1']:<15} {row['caro_label']:<15} {row['text_preview']}")

# --- DISAGREEMENT ANALYSIS — TIER 3 --------------------------------------------------------

header("TIER 3 — DISAGREEMENT ANALYSIS")

disagree_mask_t3 = gourav['tier3'] != caro['tier3']
disagree_t3 = gourav[disagree_mask_t3].copy()
disagree_t3['caro_label']   = caro.loc[disagree_mask_t3, 'tier3'].values
disagree_t3['text_preview'] = disagree_t3['video_description_english'].astype(str).str[:80]

print(f"\nTotal disagreements: {disagree_mask_t3.sum()} / {total} ({disagree_mask_t3.sum()/total*100:.1f}%)")

pairs3 = disagree_t3.groupby(['tier3', 'caro_label']).size().reset_index(name='count').sort_values('count', ascending=False)
print(f"\nMost common disagreement patterns:")
for _, row in pairs3.iterrows():
    print(f"  Gourav: {row['tier3']:<25} → Caro: {row['caro_label']:<25} ({row['count']} times)")

print(f"\nAll Tier 3 disagreements:")
print(f"{'#':<4} {'Gourav':<27} {'Caro':<27} {'Text preview'}")
print("-" * 90)
for idx, (_, row) in enumerate(disagree_t3.iterrows(), 1):
    print(f"  {idx:<3} {row['tier3']:<27} {row['caro_label']:<27} {row['text_preview']}")

# --- CONFUSION MATRIX — TIER 1 -----------------------------------------------------------

header("TIER 1 — CONFUSION MATRIX")

labels_t1_sorted = sorted(set(gourav['tier1'].unique()) | set(caro['tier1'].unique()))
cm_t1    = confusion_matrix(gourav['tier1'], caro['tier1'], labels=labels_t1_sorted)
cm_df_t1 = pd.DataFrame(cm_t1, index=labels_t1_sorted, columns=labels_t1_sorted)
cm_df_t1.index.name = "Gourav \\ Caro"
print("\nRows = Gourav labels, Columns = Caro labels")
print(cm_df_t1.to_string())

# --- CONFUSION MATRIX — TIER 3 --------------------------------------------------

header("TIER 3 — CONFUSION MATRIX")

labels_t3_sorted = sorted(set(gourav['tier3'].unique()) | set(caro['tier3'].unique()))
cm_t3    = confusion_matrix(gourav['tier3'], caro['tier3'], labels=labels_t3_sorted)
cm_df_t3 = pd.DataFrame(cm_t3, index=labels_t3_sorted, columns=labels_t3_sorted)
cm_df_t3.index.name = "Gourav \\ Caro"
print("\nRows = Gourav labels, Columns = Caro labels")
print(cm_df_t3.to_string())

# --- CLOSE FILE -------------------------------------------------------------------

sys.stdout = sys.__stdout__
log_file.close()
print(f"\nFull results saved to: {OUTPUT_FILE}")
print("--- DONE ---")
