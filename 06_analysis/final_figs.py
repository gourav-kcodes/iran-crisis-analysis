# Extra figures for the final combined report (the other figures are reused from data/analysis/report_best).
#   bench.png  : Cohen's kappa against the gold standard, every model and prompt setting, both tiers
#   cm_t1.png  : confusion matrix, Tier 1 sentiment, Mistral 7B zero-shot (400 gold rows)
#   cm_t3.png  : confusion matrix, Tier 3 frame, Mistral 7B few-shot (400 gold rows)
# Usage (project root, after analysis_core.py zs and fs): python 06_analysis/final_figs.py
import re, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

OUT = 'data/analysis/final'
import os
os.makedirs(OUT, exist_ok=True)
D = 'data/clean/'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False})
nl = lambda s: str(s).strip().replace('Humor', 'Humour')
nid = lambda s: re.sub(r'[^0-9]', '', str(s))[:18]
EZ = json.load(open('data/analysis/report_zs/results.json'))['eval']
EF = json.load(open('data/analysis/report_fs/results.json'))['eval']
ZS, FS = '#6889ad', '#b67768'

# ── benchmark ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200)
for ax, t, ttl in ((axes[0], 't1', 'Tier 1 — Sentiment'), (axes[1], 't3', 'Tier 3 — Frame')):
    models = ['Mistral', 'Qwen', 'DeepSeek'] + (['RoBERTa'] if t == 't1' else [])
    names = {'Mistral': 'Mistral 7B', 'Qwen': 'Qwen2.5-3B', 'DeepSeek': 'DeepSeek-R1\n(partial)', 'RoBERTa': 'RoBERTa\n(3-class)'}
    x = np.arange(len(models))
    for j, (E, col, lab) in enumerate(((EZ, ZS, 'Zero-shot'), (EF, FS, 'Few-shot'))):
        for i, m in enumerate(models):
            if m not in E or t not in E[m]:
                continue
            v = E[m][t]['kappa']
            ax.bar(i + (j - 0.5) * 0.36, v, 0.34, color=col, edgecolor='white', hatch='//' if m == 'DeepSeek' else None,
                   label=lab if i == 0 else None)
            ax.text(i + (j - 0.5) * 0.36, v + 0.012, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([names[m] for m in models], fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Cohen's κ vs gold standard")
    ax.set_title(ttl)
    ax.grid(True, linestyle='--', alpha=0.35, axis='y')
    ax.set_axisbelow(True)
axes[1].legend(loc='upper left', fontsize=9, frameon=True)
fig.tight_layout()
fig.savefig(f'{OUT}/bench.png', dpi=200)
plt.close(fig)

# ── confusion matrices ───────────────────────────────────────────────────────
gold = pd.read_csv(D + 'gold_standard_dataset.csv')
gold['k'] = gold['id'].map(nid)


def pred(f):
    d = pd.read_csv(D + f'iran_crisis_videos_classified_{f}.csv', low_memory=False, dtype={'id': str})
    d['k'] = d['id'].map(nid)
    for c in ('sentiment_label', 'frame_label'):
        d[c] = d[c].astype(str).map(nl)
    return gold.merge(d.drop_duplicates('k')[['k', 'sentiment_label', 'frame_label']], on='k')


SENT = ['Fear', 'Anger', 'Sadness', 'Hope', 'Solidarity', 'Frustration', 'Humour', 'Disgust', 'Curiosity', 'Neutral', 'Unclear']
FRAME = ['Political blame', 'Military/strategic', 'Economic/sanctions', 'Humanitarian', 'Religious framing', 'Satire/meme',
         'Historical context', 'Media criticism', 'Personal connection', 'Misinformation', 'Unclear']
for name, f, gcol, pcol, labs, ttl in (
        ('cm_t1', 'mistral', 'tier1_sentiment', 'sentiment_label', SENT, 'Tier 1 sentiment — Mistral 7B zero-shot'),
        ('cm_t3', 'mistral_fewshot', 'tier3_frame', 'frame_label', FRAME, 'Tier 3 frame — Mistral 7B few-shot')):
    m = pred(f)
    assert len(m) == 400
    tab = pd.crosstab(m[gcol].map(nl), m[pcol]).reindex(index=labs, columns=labs, fill_value=0)
    tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    fig, ax = plt.subplots(figsize=(11, 6.4 if name == 'cm_t3' else 5.6), dpi=200)
    sns.heatmap(tab, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=0.5, linecolor='white', cbar=False,
                annot_kws={'fontsize': 10}, vmax=max(10, np.percentile(tab.values[tab.values > 0], 90)))
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Gold-standard label')
    ax.set_title(ttl)
    ax.tick_params(axis='x', rotation=35)
    plt.setp(ax.get_xticklabels(), ha='right')
    fig.tight_layout()
    fig.savefig(f'{OUT}/{name}.png', dpi=200)
    plt.close(fig)
    print(name, tab.shape, int(tab.values.sum()), int(np.trace(tab.reindex(index=tab.index, columns=tab.index, fill_value=0).values)))
print('done')
