# Carries classification error into the label shares, using the 400-row gold standard.
# Method (calibration estimator): for each predicted label j, the gold sample gives P(true = i | predicted = j).
# The error-corrected share of label i in a group is  sum_j  q_j * P(true = i | predicted = j),  where q_j is the
# group's predicted share of j. Bootstrap (2,000 draws) resamples both the gold rows and the group's videos, so the
# interval contains classifier error as well as sampling variation. Assumption: P(true | predicted) is the same in
# every group (checked for non-English posts only indirectly, see Section 1.4 of the reports).
# Usage (project root, after analysis_core.py): python 06_analysis/classification_error.py best|zs|fs
import sys, re, json
import numpy as np, pandas as pd
from scipy import stats

S = sys.argv[1]
D = 'data/clean/'
OUT = f'data/analysis/report_{S}'
nl = lambda s: str(s).strip().replace('Humor', 'Humour')
nid = lambda s: re.sub(r'[^0-9]', '', str(s))[:18]
SENT = ['Fear', 'Anger', 'Sadness', 'Hope', 'Solidarity', 'Frustration', 'Humour', 'Disgust', 'Curiosity', 'Neutral', 'Unclear']
FRAME = ['Political blame', 'Military/strategic', 'Economic/sanctions', 'Humanitarian', 'Religious framing', 'Satire/meme',
         'Historical context', 'Media criticism', 'Personal connection', 'Misinformation', 'Unclear']
B = 2000


def load(f):
    d = pd.read_csv(D + f'iran_crisis_videos_classified_{f}.csv', low_memory=False, dtype={'id': str})
    for c in ('sentiment_label', 'frame_label'):
        d[c] = d[c].astype(str).map(nl)
    d['week'] = pd.to_datetime(d['create_time_readable'], errors='coerce').dt.to_period('W')
    d['k'] = d['id'].map(nid)
    return d


MZ, MF = load('mistral'), load('mistral_fewshot')
if S == 'best':
    M = MZ.merge(MF[['id', 'frame_label']].rename(columns={'frame_label': 'fl'}), on='id', how='left')
    M['frame_label'] = M.pop('fl')
elif S == 'zs':
    M = MZ
else:
    M = MF
gold = pd.read_csv(D + 'gold_standard_dataset.csv')
gold['k'] = gold['id'].map(nid)
gold['sentiment_true'], gold['frame_true'] = gold['tier1_sentiment'].map(nl), gold['tier3_frame'].map(nl)
G = gold.merge(M.drop_duplicates('k')[['k', 'sentiment_label', 'frame_label']], on='k')
assert len(G) == 400
rng = np.random.default_rng(42)
GIDX = rng.integers(0, 400, (B, 400))          # one set of gold resamples, shared by every estimate


def matrices(tier):
    labs = SENT if tier == 'sentiment' else FRAME
    col_p, col_t = f'{tier}_label', f'{tier}_true'
    P = pd.Categorical(G[col_p], categories=labs).codes
    T = pd.Categorical(G[col_t], categories=labs).codes
    K = len(labs)

    def mat(idx):
        c = np.zeros((K, K))
        np.add.at(c, (P[idx], T[idx]), 1)
        rows = c.sum(1)
        m = np.where(rows[:, None] > 0, c / np.maximum(rows[:, None], 1), np.eye(K))   # unseen prediction: keep it
        return m
    return labs, mat(np.arange(400)), np.stack([mat(ix) for ix in GIDX])


MAT = {t: matrices(t) for t in ('sentiment', 'frame')}


def q_boot(series, labs, seed):
    q = series.value_counts(normalize=True).reindex(labs, fill_value=0).values
    r = np.random.default_rng(seed)
    return q, r.multinomial(len(series), q, size=B) / len(series)


def corrected(series, tier, seed=1):
    labs, m0, mb = MAT[tier]
    q, qb = q_boot(series, labs, seed)
    point = q @ m0
    boot = np.einsum('bj,bji->bi', qb, mb)
    return labs, point * 100, boot * 100


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


R = {'method': 'calibration estimator, P(true|pred) from 400 gold rows, 2000 bootstrap draws'}
# overall shares
for tier, col in (('sentiment', 'sentiment_label'), ('frame', 'frame_label')):
    labs, p, bt = corrected(M[col], tier)
    R[f'overall_{tier}'] = {l: {'raw': float((M[col] == l).mean() * 100), 'corrected': float(p[i]),
                                'ci': ci(bt[:, i]), 'gold_n': int((G[f'{tier}_true'] == l).sum()),
                                'gold': float((G[f'{tier}_true'] == l).mean() * 100),
                                'gold_exact': [float(x * 100) for x in stats.binomtest(int((G[f'{tier}_true'] == l).sum()), 400).proportion_ci()]}
                            for i, l in enumerate(labs)}
# precision of each predicted label on the gold sample, with exact CI
R['precision'] = {}
for tier in ('sentiment', 'frame'):
    for l in (SENT if tier == 'sentiment' else FRAME):
        sub = G[G[f'{tier}_label'] == l]
        k = int((sub[f'{tier}_true'] == l).sum())
        R['precision'][l] = {'n': len(sub), 'k': k,
                             'ci': [float(x * 100) for x in stats.binomtest(k, len(sub)).proportion_ci()] if len(sub) else None}


def compare(name, a, b, tier, label):
    col = f'{tier}_label'
    labs, pa, ba = corrected(a[col], tier, seed=11)
    _, pb, bb = corrected(b[col], tier, seed=12)
    i = labs.index(label)
    d = ba[:, i] - bb[:, i]
    p_two = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    R['compare'][name] = {'label': label, 'raw_a': float((a[col] == label).mean() * 100), 'raw_b': float((b[col] == label).mean() * 100),
                          'corr_a': float(pa[i]), 'corr_b': float(pb[i]), 'ci_a': ci(ba[:, i]), 'ci_b': ci(bb[:, i]),
                          'diff': float(pa[i] - pb[i]), 'diff_ci': ci(d), 'p': p_two, 'na': len(a), 'nb': len(b)}


R['compare'] = {}
lang, reg = M['language_full'], M['region_code']
wk = M.dropna(subset=['week'])
cnt = wk['week'].value_counts()
weeks = sorted(cnt[cnt >= 30].index)
early, late = wk[wk['week'].isin(weeks[:2])], wk[wk['week'].isin(weeks[-2:])]
compare('Religious framing: Indonesian vs English', M[lang == 'Indonesian'], M[lang == 'English'], 'frame', 'Religious framing')
compare('Historical context: Indonesian vs English', M[lang == 'Indonesian'], M[lang == 'English'], 'frame', 'Historical context')
compare('Historical context: French vs English', M[lang == 'French'], M[lang == 'English'], 'frame', 'Historical context')
compare('Religious framing: ID vs US', M[reg == 'ID'], M[reg == 'US'], 'frame', 'Religious framing')
compare('Religious framing: NG vs US', M[reg == 'NG'], M[reg == 'US'], 'frame', 'Religious framing')
compare('Solidarity: GB vs US', M[reg == 'GB'], M[reg == 'US'], 'sentiment', 'Solidarity')
compare('Fear: NG vs US', M[reg == 'NG'], M[reg == 'US'], 'sentiment', 'Fear')
compare('Solidarity: first two weeks vs last two', early, late, 'sentiment', 'Solidarity')
compare('Humanitarian: first two weeks vs last two', early, late, 'frame', 'Humanitarian')
compare('Political blame: first two weeks vs last two', early, late, 'frame', 'Political blame')
json.dump(R, open(f'{OUT}/classification_error.json', 'w'), indent=1)
for k, v in R['compare'].items():
    print(f'{S} {k}: raw {v["raw_a"]:.1f} vs {v["raw_b"]:.1f} | corrected {v["corr_a"]:.1f} [{v["ci_a"][0]:.1f}-{v["ci_a"][1]:.1f}] vs '
          f'{v["corr_b"]:.1f} [{v["ci_b"][0]:.1f}-{v["ci_b"][1]:.1f}] | diff {v["diff"]:.1f} [{v["diff_ci"][0]:.1f}, {v["diff_ci"][1]:.1f}] p={v["p"]:.3f}')
for t in ('frame', 'sentiment'):
    print(S, t, {l: (round(v['raw'], 1), round(v['corrected'], 1), [round(x, 1) for x in v['ci']], v['gold_n']) for l, v in R[f'overall_{t}'].items()})
