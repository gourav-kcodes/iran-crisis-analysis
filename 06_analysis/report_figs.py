# Figures for the three analysis reports (best | zs | fs), drawn in the style of the Substantive report:
# short plot titles, legends outside the plotting area, same palette, at most two panels side by side.
# Usage (project root): python 06_analysis/report_figs.py best|zs|fs
#   best : Mistral zero-shot sentiment + Mistral few-shot frame; Qwen few-shot as cross-check
#   zs   : Mistral zero-shot, both tiers; Qwen zero-shot as cross-check
#   fs   : Mistral few-shot, both tiers;  Qwen few-shot as cross-check
# Output: data/analysis/report_<s>/figNN.png  (numbers match the figure numbers in the reports)
import sys, os, re, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_fscore_support

S = sys.argv[1]
OUT = f'data/analysis/report_{S}'
os.makedirs(OUT, exist_ok=True)
R = json.load(open(f'{OUT}/results.json'))
D = 'data/clean/'
MONTH_Q = {'20260301_20260331', '20260401_20260430'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False})
nl = lambda s: str(s).strip().replace('Humor', 'Humour')
nid = lambda s: re.sub(r'[^0-9]', '', str(s))[:18]


def load(f):
    d = pd.read_csv(D + f'iran_crisis_videos_classified_{f}.csv', low_memory=False, dtype={'id': str})
    for c in ('sentiment_label', 'frame_label'):
        d[c] = d[c].astype(str).map(nl)
    d['date'] = pd.to_datetime(d['create_time_readable'], errors='coerce')
    d['week'] = d['date'].dt.to_period('W')
    d['k'] = d['id'].map(nid)
    d['month_query'] = d['date_range'].isin(MONTH_Q)
    return d


MZ, MF, QZ, QF = load('mistral'), load('mistral_fewshot'), load('qwen'), load('qwen_fewshot')
DZ, DF = load('deepseek'), load('deepseek_fewshot')
if S == 'best':
    M = MZ.merge(MF[['id', 'frame_label']].rename(columns={'frame_label': 'fl'}), on='id', how='left')
    M['frame_label'] = M.pop('fl')
    Q, DSent, DFrame = QF, DZ, DF
    QN, ML = 'Qwen2.5-3B few-shot', 'Mistral 7B'
elif S == 'zs':
    M, Q, DSent, DFrame = MZ, QZ, DZ, DZ
    QN, ML = 'Qwen2.5-3B zero-shot', 'Mistral 7B'
else:
    M, Q, DSent, DFrame = MF, QF, DF, DF
    QN, ML = 'Qwen2.5-3B few-shot', 'Mistral 7B'
assert len(M) == 5848
RB = pd.read_csv(D + 'iran_crisis_videos_classified_roberta.csv', low_memory=False)
gold = pd.read_csv(D + 'gold_standard_dataset.csv')
gold['k'] = gold['id'].map(nid)
gold['gs'], gold['gf'] = gold['tier1_sentiment'].map(nl), gold['tier3_frame'].map(nl)

MB, MG, MR, MP, GREY = '#6889ad', '#85a768', '#b67768', '#9477b6', '#8a8a8a'
SENT_T = {'Fear': '#c0392b', 'Anger': '#d4813a', 'Sadness': '#3a7ca5', 'Hope': '#3a9c5b', 'Solidarity': '#8b5bb0',
          'Frustration': '#7b5040', 'Humour': '#b0407a', 'Disgust': '#5b5b8b', 'Curiosity': '#8b8b3a', 'Unclear': '#3a8b8b'}
FRAME_T = {'Political blame': '#c0392b', 'Military/strategic': '#3a7ca5', 'Economic/sanctions': '#3a9c5b',
           'Humanitarian': '#d4813a', 'Unclear': '#6b7b8b', 'Religious framing': '#7b5040', 'Satire/meme': '#b0407a',
           'Media criticism': '#5b5b8b', 'Historical context': '#8b8b3a', 'Personal connection': '#3a8b8b',
           'Misinformation': '#4a6b8b'}
SENT_B = {'Fear': '#c64d40', 'Anger': '#d88d4d', 'Sadness': '#4d89ae', 'Hope': '#4da66b', 'Solidarity': '#966bb7',
          'Frustration': '#886153', 'Humour': '#b75387', 'Disgust': '#6b6b9b', 'Curiosity': '#9b9b4b'}
FRAME_B = {'Political blame': '#6b8baf', 'Military/strategic': '#88a96b', 'Economic/sanctions': '#b7796b',
           'Humanitarian': '#d88d4d', 'Unclear': '#798896'}
MUTED = ['#6b8baf', '#88a96b', '#b7796b', '#9679b7', '#b79f6b', '#6ba7b7', '#b76b96', '#79966b', '#b7886b', '#8c8cb4', '#a0a0a0']

SENT9 = ['Fear', 'Anger', 'Sadness', 'Hope', 'Solidarity', 'Frustration', 'Humour', 'Disgust', 'Curiosity']
FR10 = ['Political blame', 'Military/strategic', 'Economic/sanctions', 'Humanitarian', 'Religious framing', 'Satire/meme',
        'Historical context', 'Media criticism', 'Personal connection', 'Misinformation']
FR11 = ['Political blame', 'Military/strategic', 'Economic/sanctions', 'Unclear', 'Humanitarian', 'Religious framing',
        'Satire/meme', 'Media criticism', 'Historical context', 'Personal connection', 'Misinformation']
LANGS = ['English', 'Indonesian', 'German', 'Spanish', 'French', 'Catalan']
REGS = ['US', 'GB', 'PK', 'NG', 'ID', 'DE', 'FR', 'AE']
SIZE = {'pc': (2136, 900), 'ci': (2316, 876), 'cif': (2496, 876), 'mod': (2166, 1056), 'modf': (2480, 1056),
        'trend': (2010, 876), 'x': (2316, 876), 'heat': (2292, 1236), 'grp': (2323, 1056), 'panel': (2200, 2700),
        'rel': (2136, 876), 'eng': (2316, 876), 'day': (2400, 1000), 'gold': (2400, 1000), 'two': (2400, 1000),
        'twoh': (2400, 1150)}
FN = [0]


def canvas(kind):
    w, h = SIZE[kind]
    return plt.figure(figsize=(w / 200, h / 200), dpi=200)


def grid(ax, axis='both'):
    ax.grid(True, linestyle='--', alpha=0.35, axis=axis)
    ax.set_axisbelow(True)


def save(fig, tight=True):
    FN[0] += 1
    if tight:
        fig.tight_layout()
    fig.savefig(f'{OUT}/fig{FN[0]:02d}.png', dpi=200)
    plt.close(fig)


def legend_right(ax, title=None, fs=9, handles=None, labels=None):
    kw = dict(title=title, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=fs, frameon=True)
    if handles:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)


def boot(mask):
    x = np.asarray(mask, float)
    rng = np.random.default_rng(42)
    p = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return x.mean() * 100, np.percentile(p, 2.5) * 100, np.percentile(p, 97.5) * 100


GM = gold.merge(M.drop_duplicates('k')[['k', 'sentiment_label', 'frame_label']], on='k')
GQ = gold.merge(Q.drop_duplicates('k')[['k', 'sentiment_label', 'frame_label']], on='k')


# ── 1, 2  per-class scores (Mistral) ─────────────────────────────────────────
def perclass(gcol, pcol, labels, title):
    a, b = GM[gcol], GM[pcol]
    P, Rr, F, Sup = precision_recall_fscore_support(a, b, labels=labels, zero_division=0)
    npred = [(b == l).sum() for l in labels]
    fig = canvas('pc')
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    for i, (v, nm, c) in enumerate(((P, 'Precision', MB), (Rr, 'Recall', MG), (F, 'F1', MR))):
        ax.bar(x + (i - 1) * 0.26, v, 0.26, label=nm, color=c, edgecolor='white')
    for i, s in enumerate(Sup):
        if s == 0:
            ax.text(i, 0.03, 'no gold\nrows', ha='center', fontsize=7, color='#555555')
    wrap = lambda l: l.replace(' ', '\n').replace('/', '/\n')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{wrap(l)}\n(gold {s},\npred {p})' for l, s, p in zip(labels, Sup, npred)], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(title)
    legend_right(ax)
    grid(ax)
    save(fig)


T1L = SENT9 + ['Neutral'] + (['Unclear'] if (GM['sentiment_label'] == 'Unclear').any() else [])
perclass('gs', 'sentiment_label', T1L, 'Per-Class Scores — Tier 1 Sentiment')
perclass('gf', 'frame_label', FR11, 'Per-Class Scores — Tier 3 Frame')

# ── 3  frame mix on the gold sample: annotators vs classifiers (same 400 videos) ──
fig = canvas('gold')
ax = fig.add_subplot(111)
x = np.arange(len(FR11))
gci = [boot(gold['gf'] == l) for l in FR11]
ax.bar(x - 0.27, [c[0] for c in gci], 0.27, color=GREY, edgecolor='white', label='Gold standard',
       yerr=[[c[0] - c[1] for c in gci], [c[2] - c[0] for c in gci]], capsize=3, error_kw={'elinewidth': 1, 'ecolor': '#333333'})
ax.bar(x, [(GM['frame_label'] == l).mean() * 100 for l in FR11], 0.27, color=MB, edgecolor='white', label=ML)
ax.bar(x + 0.27, [(GQ['frame_label'] == l).mean() * 100 for l in FR11], 0.27, color=MG, edgecolor='white', label=QN.split(' ')[0])
ax.set_xticks(x)
ax.set_xticklabels(FR11, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('% of Gold Rows')
ax.set_title('Frame Mix on the Gold Sample')
legend_right(ax)
grid(ax)
save(fig)


# ── 4, 5  bootstrap CIs ──────────────────────────────────────────────────────
def ci_fig(series, labels, kind, color, ecolor, title):
    cis = [boot(series == l) for l in labels]
    fig = canvas(kind)
    ax = fig.add_subplot(111)
    v = [c[0] for c in cis]
    ax.bar(labels, v, color=color, edgecolor='white', yerr=[[c[0] - c[1] for c in cis], [c[2] - c[0] for c in cis]],
           capsize=4, error_kw={'ecolor': ecolor, 'elinewidth': 1.2})
    for i, c in enumerate(cis):
        ax.text(i, c[2] + max(v) * 0.02, f'{c[0]:.1f}%', ha='center', fontsize=8, color=ecolor)
    ax.set_ylabel('% of Videos')
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, max(c[2] for c in cis) * 1.2)
    grid(ax)
    save(fig)


SENT_CI = sorted(SENT9, key=lambda l: -(M['sentiment_label'] == l).mean())
ci_fig(M['sentiment_label'], SENT_CI, 'ci', '#6b8baf', '#1a3a5c', 'Sentiment Distribution with 95% CIs')
FR_CI = sorted(FR10, key=lambda l: -(M['frame_label'] == l).mean())
ci_fig(M['frame_label'], FR_CI, 'cif', '#88a96b', '#2a5020', 'Frame Distribution with 95% CIs')

# ── 6, 7  distributions by model ─────────────────────────────────────────────
tag = {'best': ('zero-shot', 'few-shot', 'zero-shot'), 'zs': ('zero-shot',) * 3, 'fs': ('few-shot',) * 3}[S]
S5 = SENT9 + ['Neutral']
fig = canvas('mod')
ax = fig.add_subplot(111)
x = np.arange(len(S5))
mods = [('Mistral 7B', M, MB), ('Qwen2.5-3B', Q, MG), ('DeepSeek-R1 (partial)', DSent, MR)]
k = 3 if S == 'fs' else 4
w = 0.8 / k
for i, (nm, d, c) in enumerate(mods):
    ax.bar(x + (i - k / 2 + 0.5) * w, [(d['sentiment_label'] == l).mean() * 100 for l in S5], w, label=nm, color=c, edgecolor='white')
if S != 'fs':
    ax.bar(x[-1] + (3 - k / 2 + 0.5) * w, (RB['roberta_label'].str.lower() == 'neutral').mean() * 100, w,
           label='RoBERTa (Neutral only)', color=MP, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(S5, rotation=30, ha='right')
ax.set_ylabel('% of Videos')
ax.set_title('Sentiment Distribution by Model')
legend_right(ax, 'Model')
grid(ax)
save(fig)

fig = canvas('modf')
ax = fig.add_subplot(111)
x = np.arange(len(FR11))
for i, (nm, d, c) in enumerate([('Mistral 7B', M, MB), ('Qwen2.5-3B', Q, MG), ('DeepSeek-R1 (partial)', DFrame, MR)]):
    ax.bar(x + (i - 1) * 0.26, [(d['frame_label'] == l).mean() * 100 for l in FR11], 0.26, label=nm, color=c, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(FR11, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('% of Videos')
ax.set_title('Frame Distribution by Model')
legend_right(ax, 'Model')
grid(ax)
save(fig)

# ── 8  daily volume by collection type, with verified events ─────────────────
days = pd.date_range('2026-03-01', '2026-04-30')
dd = M.dropna(subset=['date']).assign(day=lambda d: d['date'].dt.normalize())
win = dd[~dd['month_query']].groupby('day').size().reindex(days, fill_value=0)
mon = dd[dd['month_query']].groupby('day').size().reindex(days, fill_value=0)
fig = canvas('day')
ax = fig.add_subplot(111)
ax.bar(days, win.values, 0.85, color=MB, edgecolor='white', linewidth=0.3, label='Two-day window queries')
ax.bar(days, mon.values, 0.85, bottom=win.values, color=MR, edgecolor='white', linewidth=0.3, label='Month queries')
EV = [('2026-03-25', 'US proposal\nrejected'), ('2026-04-07', 'Ceasefire\nannounced'),
      ('2026-04-12', 'Talks collapse,\nblockade'), ('2026-04-21', 'Ceasefire\nextended')]
top_y = (win + mon).max()
for dt, lab in EV:
    ax.axvline(pd.Timestamp(dt), color='#444444', linestyle=':', linewidth=1.1)
    ax.text(pd.Timestamp(dt), top_y * 1.02, lab, fontsize=7.5, ha='center', va='bottom', color='#333333')
ax.set_ylim(0, top_y * 1.22)
ax.set_xlim(days[0] - pd.Timedelta(days=1), days[-1] + pd.Timedelta(days=1))
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8)
ax.set_ylabel('Videos per Day')
ax.set_title('Videos per Day by Collection Query')
legend_right(ax, fs=8.5)
grid(ax, 'y')
save(fig)


# ── 9–18  weekly trends ──────────────────────────────────────────────────────
def weekly(df, col):
    d = df.dropna(subset=['week'])
    c = d.groupby('week').size()
    t = d.groupby('week')[col].value_counts(normalize=True).unstack(fill_value=0) * 100
    return t.loc[t.index.isin(c[c >= 30].index)]


WM_S, WQ_S = weekly(M, 'sentiment_label'), weekly(Q, 'sentiment_label')
WM_F, WQ_F = weekly(M, 'frame_label'), weekly(Q, 'frame_label')
XL = [p.start_time.strftime('%d-%m-%y') for p in WM_S.index]


def trend(t, labels, cmap, title, legend_title, ymax):
    fig = canvas('trend')
    ax = fig.add_subplot(111)
    for l in labels:
        ax.plot(range(len(XL)), t.get(l, pd.Series(0, index=t.index)).values, marker='o', markersize=4, linewidth=1.8,
                color=cmap[l], label=l)
    ax.set_xticks(range(len(XL)))
    ax.set_xticklabels(XL, rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0, ymax)
    ax.set_ylabel('% of Weekly Videos')
    ax.set_xlabel('Week starting', fontsize=9)
    ax.set_title(title)
    legend_right(ax, legend_title)
    grid(ax)
    save(fig)


def top(ts, labels):
    return np.ceil(max(t[[l for l in labels if l in t]].max().max() for t in ts) * 1.15 / 5) * 5


TS = ['Fear', 'Anger', 'Sadness', 'Hope', 'Solidarity', 'Frustration']
RS = ['Humour', 'Disgust', 'Curiosity', 'Unclear']
TF = ['Political blame', 'Military/strategic', 'Economic/sanctions', 'Humanitarian', 'Unclear']
RF = ['Religious framing', 'Satire/meme', 'Media criticism', 'Historical context', 'Personal connection', 'Misinformation']
y1, y2, y3, y4 = top([WM_S, WQ_S], TS), top([WM_S, WQ_S], RS), top([WM_F, WQ_F], TF), top([WM_F, WQ_F], RF)
trend(WM_S, TS, SENT_T, 'Sentiment Trend — Mistral 7B', 'Sentiment', y1)
trend(WQ_S, TS, SENT_T, 'Sentiment Trend — Qwen2.5-3B', 'Sentiment', y1)
trend(WM_S, RS, SENT_T, 'Remaining Sentiment — Mistral 7B', 'Sentiment', y2)
trend(WQ_S, RS, SENT_T, 'Remaining Sentiment — Qwen2.5-3B', 'Sentiment', y2)
trend(WM_F, TF, FRAME_T, 'Frame Trend — Mistral 7B', 'Frame', y3)
trend(WQ_F, TF, FRAME_T, 'Frame Trend — Qwen2.5-3B', 'Frame', y3)
trend(WM_F, RF, FRAME_T, 'Remaining Frames — Mistral 7B', 'Frame', y4)
trend(WQ_F, RF, FRAME_T, 'Remaining Frames — Qwen2.5-3B', 'Frame', y4)

for lab, cm, cq in (('Fear', '#c0392b', '#e07060'), ('Anger', '#b06b5b', '#d4813a')):
    fig = canvas('x')
    ax = fig.add_subplot(111)
    ax.plot(range(len(XL)), WM_S[lab].values, marker='o', markersize=4, linewidth=1.8, color=cm, label='Mistral 7B')
    ax.plot(range(len(XL)), WQ_S[lab].values, marker='s', markersize=4, linewidth=1.8, color=cq, linestyle='--', label='Qwen2.5-3B')
    ax.set_xticks(range(len(XL)))
    ax.set_xticklabels(XL, rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0, np.ceil(max(WM_S[lab].max(), WQ_S[lab].max()) * 1.2 / 5) * 5)
    ax.set_ylabel('% of Weekly Videos')
    ax.set_xlabel('Week starting', fontsize=9)
    ax.set_title(f'{lab} — Mistral vs Qwen')
    legend_right(ax)
    grid(ax)
    save(fig)

# ── 19–21  co-occurrence: counts, row %, lift ────────────────────────────────
SO = SENT9 + ['Neutral']
raw = pd.crosstab(M['sentiment_label'], M['frame_label']).reindex(index=SO, columns=FR11, fill_value=0)
norm = raw.div(raw.sum(axis=1), axis=0) * 100
base = M['frame_label'].value_counts(normalize=True).reindex(FR11) * 100
lift = norm / base
for tab, fmt, cmap, ttl in ((raw, 'd', 'Blues', 'Co-Occurrence — Raw Counts'), (norm, '.0f', 'YlOrBr', 'Co-Occurrence — Row %')):
    fig = canvas('heat')
    ax = fig.add_subplot(111)
    sns.heatmap(tab, annot=True, fmt=fmt, cmap=cmap, ax=ax, linewidths=0.5, linecolor='white',
                vmin=0, vmax=100 if fmt != 'd' else None, annot_kws={'size': 8})
    ax.set_title(ttl, fontsize=12)
    ax.set_xlabel('Narrative Frame', fontsize=9)
    ax.set_ylabel('Sentiment', fontsize=9)
    ax.set_xticklabels(FR11, rotation=35, ha='right', fontsize=7.5)
    ax.set_yticklabels([f'{s} (n={int(raw.loc[s].sum())})' for s in SO] if fmt != 'd' else SO, rotation=0, fontsize=7.5)
    save(fig)

fig = canvas('heat')
ax = fig.add_subplot(111)
lg = np.log2(lift.where(raw >= 5))
ann = lift.where(raw >= 5).map(lambda v: '' if pd.isna(v) else (f'{v:.0f}' if v >= 10 else f'{v:.1f}'))
sns.heatmap(lg, annot=ann, fmt='', cmap='RdBu_r', center=0, vmin=-5, vmax=5, ax=ax, linewidths=0.5, linecolor='white',
            annot_kws={'size': 8}, cbar_kws={'label': 'log2(lift)'})
ax.set_facecolor('#f2f2f2')
ax.set_title('Co-Occurrence — Lift', fontsize=12)
ax.set_xlabel('Narrative Frame', fontsize=9)
ax.set_ylabel('Sentiment', fontsize=9)
ax.set_xticklabels(FR11, rotation=35, ha='right', fontsize=7.5)
ax.set_yticklabels([f'{s} (n={int(raw.loc[s].sum())})' for s in SO], rotation=0, fontsize=7.5)
save(fig)


# ── 22–30  language and region ───────────────────────────────────────────────
def shares(df, key, groups, col):
    return {g: df[df[key] == g][col].value_counts(normalize=True) * 100 for g in groups}


def grouped(groups, key, col, labels, cmap, title, legend_title):
    sm = shares(M, key, groups, col)
    cnt = M[key].value_counts()
    fig = canvas('grp')
    ax = fig.add_subplot(111)
    x = np.arange(len(groups))
    w = 0.84 / len(labels)
    for j, l in enumerate(labels):
        ax.bar(x + (j - len(labels) / 2 + 0.5) * w, [sm[g].get(l, 0) for g in groups], w, color=cmap[l], edgecolor='white', label=l)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}\n({cnt[g]})' for g in groups], fontsize=9)
    ax.set_ylim(0, max(sm[g].get(l, 0) for g in groups for l in labels) * 1.15)
    ax.set_ylabel('% of Videos')
    ax.set_title(title)
    legend_right(ax, legend_title)
    grid(ax)
    save(fig)


def panel_grid(groups, key, title):
    labels = ['Fear', 'Anger', 'Sadness', 'Hope', 'Solidarity', 'Frustration', 'Humour', 'Curiosity']
    sm, sq = shares(M, key, groups, 'sentiment_label'), shares(Q, key, groups, 'sentiment_label')
    cnt = M[key].value_counts()
    fig = canvas('panel')
    axes = fig.subplots(4, 2)
    x = np.arange(len(groups))
    for ax, l in zip(axes.flat, labels):
        ax.bar(x - 0.2, [sm[g].get(l, 0) for g in groups], 0.4, color=SENT_B[l], edgecolor='white')
        ax.bar(x + 0.2, [sq[g].get(l, 0) for g in groups], 0.4, color=SENT_B[l], alpha=0.45, hatch='//', edgecolor='white')
        ax.set_title(l, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{g[:3] if len(g) > 3 else g}\n({cnt[g]})' for g in groups], fontsize=7.5)
        ax.set_ylabel('% of Videos', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        grid(ax)
    fig.legend([Patch(facecolor='#6b6b6b', edgecolor='white'), Patch(facecolor='#c8c8c8', hatch='//', edgecolor='white')],
               ['Mistral 7B (solid)', 'Qwen2.5-3B (hatched)'], loc='lower center', ncol=2, fontsize=10, frameon=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    save(fig, tight=False)


def ci_groups(ax, groups, key, lab, color, title):
    cis = {g: boot(M[M[key] == g]['frame_label'] == lab) for g in groups}
    cnt = M[key].value_counts()
    v = [cis[g][0] for g in groups]
    ax.bar(range(len(groups)), v, color=color, edgecolor='white',
           yerr=[[cis[g][0] - cis[g][1] for g in groups], [cis[g][2] - cis[g][0] for g in groups]], capsize=4,
           error_kw={'ecolor': '#222222', 'elinewidth': 1.2})
    hi = max(c[2] for c in cis.values())
    for i, g in enumerate(groups):
        ax.text(i, cis[g][2] + hi * 0.02, f'{cis[g][0]:.1f}%', ha='center', fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f'{g}\n(n={cnt[g]})' for g in groups], fontsize=8.5)
    ax.set_ylim(0, hi * 1.25)
    ax.set_ylabel('% of Videos')
    ax.set_title(title)
    grid(ax)


def relig(groups, key, color, title):
    fig = canvas('rel')
    ci_groups(fig.add_subplot(111), groups, key, 'Religious framing', color, title)
    save(fig)


TS7 = TS + ['Humour']
grouped(LANGS, 'language_full', 'sentiment_label', TS7, SENT_B, 'Sentiment by Language — Mistral 7B', 'Sentiment')
panel_grid(LANGS, 'language_full', 'Sentiment by Language — Mistral vs Qwen')
grouped(LANGS, 'language_full', 'frame_label', TF, FRAME_B, 'Frame by Language — Mistral 7B', 'Frame')
relig(LANGS, 'language_full', '#886153', 'Religious Framing by Language with 95% CIs')
grouped(REGS, 'region_code', 'sentiment_label', TS7, SENT_B, 'Sentiment by Region — Mistral 7B', 'Sentiment')
panel_grid(REGS, 'region_code', 'Sentiment by Region — Mistral vs Qwen')
grouped(REGS, 'region_code', 'frame_label', TF, FRAME_B, 'Frame by Region — Mistral 7B', 'Frame')
relig(REGS, 'region_code', '#6b8baf', 'Religious Framing by Region with 95% CIs')
fig = canvas('twoh')
a1, a2 = fig.subplots(1, 2, gridspec_kw={'width_ratios': [6, 8]})
ci_groups(a1, LANGS, 'language_full', 'Historical context', '#9b9b4b', 'By Language')
ci_groups(a2, REGS, 'region_code', 'Historical context', '#8b8b3a', 'By Region')
plt.setp(a1.get_xticklabels(), fontsize=7.5)
plt.setp(a2.get_xticklabels(), fontsize=7.5)
fig.suptitle('Historical Context Framing with 95% CIs', fontsize=13, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, tight=False)

# ── 31–36  engagement (Unclear sentiment excluded) ───────────────────────────
SE = SENT9 + ['Neutral']
ENG = {'sentiment_label': R['eng_sent_m'], 'frame_label': R['eng_frame_m']}


def eng(col, labels, metric, ylabel, title):
    e = ENG[col]
    key = 'likes_ci' if metric == 'likes' else 'views_ci'
    med = [e[l][key][0] for l in labels]
    lo = [e[l][key][0] - e[l][key][1] for l in labels]
    hi = [e[l][key][2] - e[l][key][0] for l in labels]
    fig = canvas('eng')
    ax = fig.add_subplot(111)
    ax.bar(range(len(labels)), med, color=MUTED[:len(labels)], edgecolor='white', yerr=[lo, hi], capsize=3,
           error_kw={'ecolor': '#444444', 'elinewidth': 1})
    up = max(m + h for m, h in zip(med, hi))
    for i, (m, h) in enumerate(zip(med, hi)):
        ax.text(i + 0.18, m + up * 0.015, f'{m:g}', ha='left', fontsize=7.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f'{l}\n(n={e[l]["n"]:,})' for l in labels], rotation=30, ha='right', fontsize=7.5)
    ax.set_ylim(0, up * 1.12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    grid(ax, 'y')
    save(fig)


eng('sentiment_label', SE, 'likes', 'Median Likes', 'Median Likes by Sentiment')
eng('sentiment_label', SE, 'views', 'Median Views', 'Median Views by Sentiment')
eng('frame_label', FR11, 'likes', 'Median Likes', 'Median Likes by Frame')
eng('frame_label', FR11, 'views', 'Median Views', 'Median Views by Frame')


def means(col, labels, title):
    e = ENG[col]
    fig = canvas('two')
    a1, a2 = fig.subplots(1, 2, sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, met, nm, c in ((a1, 'mean_shares', 'Mean Shares', MB), (a2, 'mean_comments', 'Mean Comments', MR)):
        v = [e[l][met] for l in labels]
        ax.barh(y, v, color=c, edgecolor='white')
        for yi, vi in zip(y, v):
            ax.text(vi + max(v) * 0.015, yi, f'{vi:.1f}', va='center', fontsize=7.5)
        ax.set_xlim(0, max(v) * 1.18)
        ax.set_title(nm, fontsize=11)
        grid(ax, 'x')
    a1.set_yticks(y)
    a1.set_yticklabels([f'{l} (n={e[l]["n"]:,})' for l in labels], fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, tight=False)


means('sentiment_label', SE, 'Mean Shares and Comments by Sentiment')
means('frame_label', FR11, 'Mean Shares and Comments by Frame')
print('figures written to', OUT, FN[0])
