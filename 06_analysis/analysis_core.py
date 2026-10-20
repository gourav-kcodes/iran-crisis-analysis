# Shared analysis for the three analysis reports.
#   best : Mistral zero-shot sentiment + Mistral few-shot frame (Substantive report); Qwen few-shot as cross-check
#   zs   : Mistral zero-shot, both tiers; Qwen zero-shot as cross-check
#   fs   : Mistral few-shot, both tiers;  Qwen few-shot as cross-check
# Usage (project root): python 06_analysis/analysis_core.py best|zs|fs   -> data/analysis/report_<s>/results.json
import sys, os, re, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

S = sys.argv[1]
OUT = f'data/analysis/report_{S}'
os.makedirs(OUT, exist_ok=True)
D = 'data/clean/'
SENT = ['Fear', 'Anger', 'Sadness', 'Hope', 'Solidarity', 'Frustration', 'Humour', 'Disgust', 'Curiosity', 'Neutral', 'Unclear']
EMO = SENT[:9]
FRAME = ['Political blame', 'Military/strategic', 'Economic/sanctions', 'Humanitarian', 'Religious framing', 'Satire/meme',
         'Historical context', 'Media criticism', 'Personal connection', 'Misinformation', 'Unclear']
LANGS = ['English', 'Indonesian', 'German', 'Spanish', 'French', 'Catalan']
REGS = ['US', 'GB', 'PK', 'NG', 'ID', 'AE', 'FR', 'DE']
MONTH_Q = {'20260301_20260331', '20260401_20260430'}
nid = lambda s: re.sub(r'[^0-9]', '', str(s))[:18]
nl = lambda s: str(s).strip().replace('Humor', 'Humour')


def load(f):
    d = pd.read_csv(D + f, low_memory=False, dtype={'id': str})
    for c in ('sentiment_label', 'frame_label'):
        d[c] = d[c].astype(str).map(nl)
    d['date'] = pd.to_datetime(d['create_time_readable'], errors='coerce')
    d['week'] = d['date'].dt.to_period('W')
    d['k'] = d['id'].map(nid)
    d['month_query'] = d['date_range'].isin(MONTH_Q)
    return d


f = lambda n: f'iran_crisis_videos_classified_{n}.csv'
MZ, MF, QZ, QF = load(f('mistral')), load(f('mistral_fewshot')), load(f('qwen')), load(f('qwen_fewshot'))
DZ, DF = load(f('deepseek')), load(f('deepseek_fewshot'))


def combine(sent_df, frame_df):
    m = sent_df.merge(frame_df[['id', 'frame_label']].rename(columns={'frame_label': 'fl'}), on='id', how='left')
    m['frame_label'] = m['fl']
    return m.drop(columns='fl')


if S == 'best':
    M = combine(MZ, MF)
    Q = QF
    DSent, DFrame = DZ, DF
elif S == 'zs':
    M, Q, DSent, DFrame = MZ, QZ, DZ, DZ
else:
    M, Q, DSent, DFrame = MF, QF, DF, DF
assert len(M) == 5848 and M['frame_label'].notna().all()
R = {'setting': S, 'n': len(M), 'n_no_date': int(M['date'].isna().sum()), 'n_dup_id': int(M['id'].duplicated().sum()),
     'n_deepseek_sent': len(DSent), 'n_deepseek_frame': len(DFrame)}
RB = pd.read_csv(D + 'iran_crisis_videos_classified_roberta.csv', low_memory=False)
R['roberta_dist'] = (RB['roberta_label'].str.capitalize().value_counts(normalize=True) * 100).to_dict()


def boot(mask, seed=42):
    x = np.asarray(mask, float)
    if len(x) == 0:
        return [None] * 3
    rng = np.random.default_rng(seed)
    p = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return [x.mean() * 100, np.percentile(p, 2.5) * 100, np.percentile(p, 97.5) * 100]


# ── collection design ─────────────────────────────────────────────────────────
R['collection'] = {'month_query_n': int(M['month_query'].sum()),
                   'window_n': int((~M['month_query']).sum()),
                   'by_range': M['date_range'].value_counts().to_dict()}
day = M.dropna(subset=['date']).assign(day=lambda d: d['date'].dt.date.astype(str))
R['daily'] = {k: {'window': int(v[~v['month_query']].shape[0]), 'month': int(v[v['month_query']].shape[0])}
              for k, v in day.groupby('day')}

# ── reliability against gold ─────────────────────────────────────────────────
gold = pd.read_csv(D + 'gold_standard_dataset.csv')
gold['k'] = gold['id'].map(nid)
gold['gs'], gold['gf'] = gold['tier1_sentiment'].map(nl), gold['tier3_frame'].map(nl)
gold['en'] = gold['language_full'] == 'English'


def gold_merge(sent_df, frame_df):
    a = gold.merge(sent_df.drop_duplicates('k')[['k', 'sentiment_label']], on='k')
    b = gold.merge(frame_df.drop_duplicates('k')[['k', 'frame_label']], on='k')
    return a, b


def evaluate(sent_df, frame_df):
    a, b = gold_merge(sent_df, frame_df)
    out = {'n_t1': len(a), 'n_t3': len(b)}
    pcs = {}
    for tier, df, g, p, labs in (('t1', a, 'gs', 'sentiment_label', SENT), ('t3', b, 'gf', 'frame_label', FRAME)):
        out[tier] = {'kappa': cohen_kappa_score(df[g], df[p]), 'wf1': f1_score(df[g], df[p], average='weighted', zero_division=0),
                     'mf1': f1_score(df[g], df[p], average='macro', zero_division=0), 'agree': (df[g] == df[p]).mean() * 100}
        P, Rc, F, Sp = precision_recall_fscore_support(df[g], df[p], labels=labs, zero_division=0)
        pcs[tier] = {l: {'p': P[i], 'r': Rc[i], 'f1': F[i], 'support': int(Sp[i]), 'npred': int((df[p] == l).sum())}
                     for i, l in enumerate(labs)}
    return out, pcs, a, b


R['eval'], R['perclass'] = {}, {}
R['eval']['Mistral'], R['perclass']['Mistral'], GA, GB = evaluate(M, M)
R['eval']['Qwen'], R['perclass']['Qwen'], _, _ = evaluate(Q, Q)
R['eval']['DeepSeek'], _, _, _ = evaluate(DSent, DFrame)
if S != 'fs':
    mr = gold.merge(RB.assign(k=RB['id'].map(nid)).drop_duplicates('k')[['k', 'roberta_label']], on='k')
    g3 = mr['gs'].map(lambda l: 'Positive' if l in ('Humour', 'Hope', 'Solidarity') else 'Negative' if l in (
        'Anger', 'Fear', 'Sadness', 'Disgust', 'Frustration', 'Curiosity') else 'Neutral')
    rp = mr['roberta_label'].str.lower().map({'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}).fillna('Neutral')
    R['eval']['RoBERTa'] = {'n_t1': len(mr), 't1': {'kappa': cohen_kappa_score(g3, rp), 'wf1': f1_score(g3, rp, average='weighted'),
                                                     'mf1': f1_score(g3, rp, average='macro'), 'agree': (g3 == rp).mean() * 100}}
R['gold_counts'] = {'t1': gold['gs'].value_counts().to_dict(), 't3': gold['gf'].value_counts().to_dict()}
R['gold_frame_ci'] = {l: boot(gold['gf'] == l) for l in FRAME}
R['gold_neutral_ci'] = boot(gold['gs'] == 'Neutral')
R['gold_emotional'] = float((~gold['gs'].isin(['Neutral', 'Unclear'])).mean() * 100)
R['pred_emotional_gold'] = float((~GA['sentiment_label'].isin(['Neutral', 'Unclear'])).mean() * 100)
R['fear_pred_gold_labels'] = GA[GA['sentiment_label'] == 'Fear']['gs'].value_counts().to_dict()
R['mil_as_pb'] = int(((GB['gf'] == 'Military/strategic') & (GB['frame_label'] == 'Political blame')).sum())

# ── distributions and CIs ────────────────────────────────────────────────────
R['dist'] = {'Mistral': {'sent': (M['sentiment_label'].value_counts(normalize=True) * 100).to_dict(),
                         'frame': (M['frame_label'].value_counts(normalize=True) * 100).to_dict()},
             'Qwen': {'sent': (Q['sentiment_label'].value_counts(normalize=True) * 100).to_dict(),
                      'frame': (Q['frame_label'].value_counts(normalize=True) * 100).to_dict()},
             'DeepSeek': {'sent': (DSent['sentiment_label'].value_counts(normalize=True) * 100).to_dict(),
                          'frame': (DFrame['frame_label'].value_counts(normalize=True) * 100).to_dict()}}
R['counts'] = {'sent': M['sentiment_label'].value_counts().to_dict(), 'frame': M['frame_label'].value_counts().to_dict()}
R['ci_sent'] = {l: boot(M['sentiment_label'] == l) for l in SENT}
R['ci_frame'] = {l: boot(M['frame_label'] == l) for l in FRAME}

# ── non-English ──────────────────────────────────────────────────────────────
en = M['language_full'] == 'English'
txt = M['video_description_english'].fillna('').astype(str).str.strip()
short = (txt.str.len() < 10) | (txt.str.split().str.len() < 4)
R['short'] = {'n': int(short.sum()), 'en_share': float(short[en].mean() * 100), 'ne_share': float(short[~en].mean() * 100),
              'short_unclear_frame': int((short & (M['frame_label'] == 'Unclear')).sum()),
              'short_unclear_sent': int((short & (M['sentiment_label'] == 'Unclear')).sum())}
R['nonenglish'] = {}
for r in REGS:
    s = M[M['region_code'] == r]
    e, n = s[s['language_full'] == 'English'], s[s['language_full'] != 'English']
    R['nonenglish'][r] = {'n': len(s), 'n_en': len(e), 'n_ne': len(n),
                          'unc_all': (s['frame_label'] == 'Unclear').mean() * 100, 'unc_en': (e['frame_label'] == 'Unclear').mean() * 100,
                          'unc_ne': (n['frame_label'] == 'Unclear').mean() * 100,
                          'neu_all': (s['sentiment_label'] == 'Neutral').mean() * 100, 'neu_en': (e['sentiment_label'] == 'Neutral').mean() * 100,
                          'neu_ne': (n['sentiment_label'] == 'Neutral').mean() * 100}
R['ne_overall'] = {'unc_en': boot(M[en]['frame_label'] == 'Unclear'), 'unc_ne': boot(M[~en]['frame_label'] == 'Unclear'),
                   'neu_en': boot(M[en]['sentiment_label'] == 'Neutral'), 'neu_ne': boot(M[~en]['sentiment_label'] == 'Neutral'),
                   'n_en': int(en.sum()), 'n_ne': int((~en).sum())}
NEb, ENb = GB[~GB['en']], GB[GB['en']]
NEa, ENa = GA[~GA['en']], GA[GA['en']]
R['ne_gold'] = {'n_ne': len(NEb), 'n_en': len(ENb),
                'gold_unc_ne': (NEb['gf'] == 'Unclear').mean() * 100, 'gold_unc_en': (ENb['gf'] == 'Unclear').mean() * 100,
                'pred_unc_ne': (NEb['frame_label'] == 'Unclear').mean() * 100,
                'prec_unc_ne': ((NEb['frame_label'] == 'Unclear') & (NEb['gf'] == 'Unclear')).sum() / max(1, (NEb['frame_label'] == 'Unclear').sum()) * 100,
                'n_framed_ne': int((NEb['gf'] != 'Unclear').sum()), 'n_framed_en': int((ENb['gf'] != 'Unclear').sum()),
                'agree_framed_ne': (NEb[NEb['gf'] != 'Unclear']['frame_label'] == NEb[NEb['gf'] != 'Unclear']['gf']).mean() * 100,
                'agree_framed_en': (ENb[ENb['gf'] != 'Unclear']['frame_label'] == ENb[ENb['gf'] != 'Unclear']['gf']).mean() * 100,
                'gold_neu_ne': (NEa['gs'] == 'Neutral').mean() * 100, 'gold_neu_en': (ENa['gs'] == 'Neutral').mean() * 100,
                'pred_neu_ne': (NEa['sentiment_label'] == 'Neutral').mean() * 100, 'pred_neu_en': (ENa['sentiment_label'] == 'Neutral').mean() * 100}


# ── time ─────────────────────────────────────────────────────────────────────
def wk_label(p):
    return f"{p.start_time.strftime('%d %b')} - {p.end_time.strftime('%d %b')}"


def weekly(df, col, drop_month=False):
    d = df.dropna(subset=['week'])
    if drop_month:
        d = d[~d['month_query']]
    c = d.groupby('week').size()
    t = d.groupby('week')[col].value_counts(normalize=True).unstack(fill_value=0) * 100
    cnt = d.groupby('week')[col].value_counts().unstack(fill_value=0)
    keep = c[c >= 30].index
    return t.loc[t.index.isin(keep)], cnt.loc[cnt.index.isin(keep)], c.loc[c.index.isin(keep)]


R['weekly'] = {}
for tag, df, col in (('sent_m', M, 'sentiment_label'), ('frame_m', M, 'frame_label'), ('sent_q', Q, 'sentiment_label'), ('frame_q', Q, 'frame_label')):
    t, cnt, tot = weekly(df, col)
    R['weekly'][tag] = {c: {wk_label(k): float(v) for k, v in t[c].items()} for c in t.columns}
    R['weekly'][tag + '_count'] = {c: {wk_label(k): int(v) for k, v in cnt[c].items()} for c in cnt.columns}
    t2, _, _ = weekly(df, col, True)
    R['weekly'][tag + '_nomonth'] = {c: {wk_label(k): float(v) for k, v in t2[c].items()} for c in t2.columns}
_, _, tot = weekly(M, 'sentiment_label')
R['week_totals'] = {wk_label(k): int(v) for k, v in tot.items()}
_, _, tot2 = weekly(M, 'sentiment_label', True)
R['week_totals_nomonth'] = {wk_label(k): int(v) for k, v in tot2.items()}


def armitage(df, col, label, drop_month=False):
    d = df.dropna(subset=['week'])
    if drop_month:
        d = d[~d['month_query']]
    c = d.groupby('week').size()
    d = d[d['week'].isin(c[c >= 30].index)]
    weeks = sorted(d['week'].unique())
    t = np.arange(len(weeks))
    n = np.array([(d['week'] == w).sum() for w in weeks])
    x = np.array([((d['week'] == w) & (d[col] == label)).sum() for w in weeks])
    N, X = n.sum(), x.sum()
    p = X / N
    T = (t * x).sum() - p * (t * n).sum()
    var = p * (1 - p) * ((t * t * n).sum() - (t * n).sum() ** 2 / N)
    z = T / np.sqrt(var)
    return {'z': float(z), 'p': float(2 * stats.norm.sf(abs(z)))}


R['trend_tests'] = {}
for nm, df, col, lab in (('Solidarity_m', M, 'sentiment_label', 'Solidarity'), ('Solidarity_q', Q, 'sentiment_label', 'Solidarity'),
                         ('Humanitarian_m', M, 'frame_label', 'Humanitarian'), ('Humanitarian_q', Q, 'frame_label', 'Humanitarian'),
                         ('Humour_m', M, 'sentiment_label', 'Humour'), ('Satire_m', M, 'frame_label', 'Satire/meme'),
                         ('Fear_m', M, 'sentiment_label', 'Fear'), ('Anger_m', M, 'sentiment_label', 'Anger'),
                         ('Political_m', M, 'frame_label', 'Political blame'), ('Historical_m', M, 'frame_label', 'Historical context')):
    R['trend_tests'][nm] = armitage(df, col, lab)
    R['trend_tests'][nm + '_nomonth'] = armitage(df, col, lab, True)


def wcorr(tag_a, tag_b, lab):
    a = pd.Series(R['weekly'][tag_a].get(lab, {}))
    b = pd.Series(R['weekly'][tag_b].get(lab, {})).reindex(a.index)
    return float(np.corrcoef(a.values, b.values)[0, 1])


R['week_corr'] = {'Fear': wcorr('sent_m', 'sent_q', 'Fear'), 'Anger': wcorr('sent_m', 'sent_q', 'Anger'),
                  'Political blame': wcorr('frame_m', 'frame_q', 'Political blame'),
                  'Solidarity': wcorr('sent_m', 'sent_q', 'Solidarity')}

# ── co-occurrence with lift and Fisher tests ─────────────────────────────────


def cooc(df):
    out = {}
    base = df['frame_label'].value_counts(normalize=True)
    for s in SENT:
        sub = df[df['sentiment_label'] == s]
        if len(sub) == 0:
            continue
        sh = sub['frame_label'].value_counts(normalize=True)
        row = {'n': len(sub)}
        for fr in FRAME:
            a = int((sub['frame_label'] == fr).sum())
            b = len(sub) - a
            c = int(((df['sentiment_label'] != s) & (df['frame_label'] == fr)).sum())
            d_ = int(((df['sentiment_label'] != s) & (df['frame_label'] != fr)).sum())
            row[fr] = {'pct': float(sh.get(fr, 0) * 100), 'lift': float(sh.get(fr, 0) / base.get(fr, np.nan)) if base.get(fr, 0) else None,
                       'p': float(stats.fisher_exact([[a, b], [c, d_]], alternative='greater')[1]) if a > 0 else 1.0}
        out[s] = row
    return out


R['cooc_m'], R['cooc_q'] = cooc(M), cooc(Q)
R['cooc_ci'] = {}
for s in EMO:
    sub = M[M['sentiment_label'] == s]
    tops = sorted([fr for fr in FRAME if fr != 'Unclear'], key=lambda fr: -(sub['frame_label'] == fr).mean())
    R['cooc_ci'][s] = {fr: boot(sub['frame_label'] == fr) for fr in tops[:2]}

# ── language and region ──────────────────────────────────────────────────────


def shares(df, key, groups, col):
    return {g: (df[df[key] == g][col].value_counts(normalize=True) * 100).to_dict() for g in groups}


for key, groups, nm in (('language_full', LANGS, 'lang'), ('region_code', REGS, 'reg')):
    R[f'{nm}_n'] = {g: int((M[key] == g).sum()) for g in groups}
    R[f'{nm}_sent_m'], R[f'{nm}_frame_m'] = shares(M, key, groups, 'sentiment_label'), shares(M, key, groups, 'frame_label')
    R[f'{nm}_sent_q'], R[f'{nm}_frame_q'] = shares(Q, key, groups, 'sentiment_label'), shares(Q, key, groups, 'frame_label')
    sub = M[M[key].isin(groups)]
    R[f'{nm}_chi'] = {}
    for col in ('sentiment_label', 'frame_label'):
        tab = pd.crosstab(sub[key], sub[col])
        tab = tab.loc[:, tab.sum() >= 5]
        chi, p, dof, _ = stats.chi2_contingency(tab)
        V = np.sqrt(chi / (tab.values.sum() * (min(tab.shape) - 1)))
        R[f'{nm}_chi'][col] = {'chi2': float(chi), 'df': int(dof), 'p': float(p), 'V': float(V)}
    R[f'{nm}_ci'] = {}
    for lab, col in (('Religious framing', 'frame_label'), ('Historical context', 'frame_label'), ('Fear', 'sentiment_label'),
                     ('Anger', 'sentiment_label'), ('Solidarity', 'sentiment_label'), ('Unclear', 'frame_label')):
        R[f'{nm}_ci'][lab] = {g: boot(M[M[key] == g][col] == lab) for g in groups}
    R[f'{nm}_ci']['Fear+Anger'] = {g: boot(M[M[key] == g]['sentiment_label'].isin(['Fear', 'Anger'])) for g in groups}
vc = M['region_code'].value_counts()
R['reg50'] = {r: {'n': int(vc[r]), 's': (M[M['region_code'] == r]['sentiment_label'].value_counts(normalize=True) * 100).to_dict(),
                  'f': (M[M['region_code'] == r]['frame_label'].value_counts(normalize=True) * 100).to_dict()} for r in vc[vc >= 50].index}
vl = M['language_full'].value_counts()
R['lang50'] = {l: {'n': int(vl[l]), 's': (M[M['language_full'] == l]['sentiment_label'].value_counts(normalize=True) * 100).to_dict(),
                   'f': (M[M['language_full'] == l]['frame_label'].value_counts(normalize=True) * 100).to_dict()} for l in vl[vl >= 50].index}


def fisher(key, ga, gb, col, lab):
    A, B = M[M[key] == ga], M[M[key] == gb]
    a, b = int((A[col] == lab).sum()), int((B[col] == lab).sum())
    OR, p = stats.fisher_exact([[a, len(A) - a], [b, len(B) - b]])
    return {'a': a, 'na': len(A), 'b': b, 'nb': len(B), 'pa': a / len(A) * 100, 'pb': b / len(B) * 100, 'p': float(p)}


R['pair_tests'] = {}
for key, ga, gb, col, lab in (
        ('language_full', 'Indonesian', 'English', 'frame_label', 'Religious framing'),
        ('language_full', 'French', 'English', 'sentiment_label', 'Fear'),
        ('language_full', 'French', 'English', 'sentiment_label', 'Anger'),
        ('language_full', 'French', 'English', 'frame_label', 'Historical context'),
        ('language_full', 'Indonesian', 'English', 'frame_label', 'Historical context'),
        ('language_full', 'Spanish', 'English', 'frame_label', 'Misinformation'),
        ('region_code', 'ID', 'US', 'frame_label', 'Religious framing'),
        ('region_code', 'NG', 'US', 'frame_label', 'Religious framing'),
        ('region_code', 'ID', 'US', 'frame_label', 'Historical context'),
        ('region_code', 'US', 'DE', 'sentiment_label', 'Fear'),
        ('region_code', 'NG', 'US', 'sentiment_label', 'Fear'),
        ('region_code', 'GB', 'US', 'sentiment_label', 'Solidarity'),
        ('region_code', 'AE', 'US', 'sentiment_label', 'Anger')):
    R['pair_tests'][f'{ga}|{gb}|{lab}'] = fisher(key, ga, gb, col, lab)

# ── engagement ───────────────────────────────────────────────────────────────


def bmed(x, seed=42):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.median(x[rng.integers(0, len(x), (2000, len(x)))], axis=1)
    return [float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def eng(df, col, labels):
    out = {}
    for l in labels:
        s = df[df[col] == l]
        if len(s) == 0:
            continue
        rest = df[(df[col] != l) & (df[col] != 'Unclear' if col == 'sentiment_label' else df[col].notna())]
        out[l] = {'n': len(s), 'med_likes': float(s['like_count'].median()), 'med_views': float(s['view_count'].median()),
                  'mean_shares': float(s['share_count'].mean()), 'med_shares': float(s['share_count'].median()),
                  'mean_comments': float(s['comment_count'].mean()), 'mean_likes': float(s['like_count'].mean()),
                  'mean_views': float(s['view_count'].mean()),
                  'likes_ci': bmed(s['like_count']), 'views_ci': bmed(s['view_count']),
                  'mw_likes_p': float(stats.mannwhitneyu(s['like_count'], rest['like_count']).pvalue),
                  'mw_views_p': float(stats.mannwhitneyu(s['view_count'], rest['view_count']).pvalue),
                  'mw_likes_dir': 'higher' if s['like_count'].median() > rest['like_count'].median() else 'lower',
                  'mw_views_dir': 'higher' if s['view_count'].median() > rest['view_count'].median() else 'lower'}
    return out


SE = [l for l in SENT if l != 'Unclear']
R['eng_sent_m'], R['eng_frame_m'] = eng(M, 'sentiment_label', SE), eng(M, 'frame_label', FRAME)
R['eng_sent_q'], R['eng_frame_q'] = eng(Q, 'sentiment_label', SE), eng(Q, 'frame_label', FRAME)
R['kw'] = {}
for nm, col, labs in (('sent', 'sentiment_label', SE), ('frame', 'frame_label', FRAME)):
    for met in ('like_count', 'view_count'):
        groups = [M[M[col] == l][met].values for l in labs if (M[col] == l).sum() > 0]
        h, p = stats.kruskal(*groups)
        R['kw'][f'{nm}_{met}'] = {'H': float(h), 'p': float(p)}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


json.dump(clean(R), open(f'{OUT}/results.json', 'w'), indent=1)
print('done', S)
