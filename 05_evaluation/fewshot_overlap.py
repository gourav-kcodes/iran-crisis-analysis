# How many few-shot examples are gold-standard rows, and how the few-shot scores change without those rows.
# Reads the examples from 04_classification/few_shot_examples.py (kept out of the repository, see the template there).
# All three models share the same 42 examples except the two Fear examples, which were written for the prompt and
# are not gold rows, so the overlap is the same for every model.
# Usage (project root): python 05_evaluation/fewshot_overlap.py  -> data/analysis/final/fewshot_overlap.json
import re, json, runpy
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

ex_file = runpy.run_path('04_classification/few_shot_examples.py')
S, F = ex_file['FEW_SHOT_SENTIMENT'], ex_file['FEW_SHOT_FRAME']
norm = lambda s: re.sub(r'\W+', '', str(s).lower())
nid = lambda s: re.sub(r'[^0-9]', '', str(s))[:18]
nl = lambda s: str(s).strip().replace('Humor', 'Humour')
# An example counts as taken from a gold row when its text matches the row exactly or is an excerpt of it
# (most examples are the first sentence of a gold description), or when it is a light edit of the row
# (character similarity >= 0.75 on the shorter text).
import difflib
EX = [x for D in (S, F) for v in D.values() for x in v]
sp = lambda s: re.sub(r'\W+', ' ', str(s).lower()).strip()
g = pd.read_csv('data/clean/gold_standard_dataset.csv')
texts = [(sp(a), sp(b)) for a, b in zip(g['video_description_english'], g['video_description_clean'])]


def source_rows(x):
    x = sp(x)
    hit = {i for i, (a, b) in enumerate(texts) if x and (x == a or x == b or x in a or x in b)}
    if not hit:
        for i, (a, b) in enumerate(texts):
            for t in (a, b):
                n = min(len(x), len(t))
                if n and difflib.SequenceMatcher(None, x, t[:max(len(x), 1) + 20]).ratio() >= 0.75:
                    hit.add(i)
    return hit


src_rows = [source_rows(x) for x in EX]
rows = set().union(*src_rows)
g['ov'] = g.index.isin(rows)
n_from_gold = sum(1 for h in src_rows if h)
g['k'] = g['id'].map(nid)
out = {'n_examples': sum(map(len, S.values())) + sum(map(len, F.values())),
       'n_sent_examples': sum(map(len, S.values())), 'n_frame_examples': sum(map(len, F.values())),
       'n_examples_from_gold': n_from_gold, 'n_gold_overlap': int(g['ov'].sum()), 'scores': {}}
for f in ('mistral_fewshot', 'qwen_fewshot'):
    d = pd.read_csv(f'data/clean/iran_crisis_videos_classified_{f}.csv', low_memory=False, dtype={'id': str})
    d['k'] = d['id'].map(nid)
    m = g.merge(d.drop_duplicates('k')[['k', 'sentiment_label', 'frame_label']], on='k')
    assert len(m) == 400
    for t, gc, pc in (('t1', 'tier1_sentiment', 'sentiment_label'), ('t3', 'tier3_frame', 'frame_label')):
        a, b, keep = m[gc].map(nl), m[pc].map(nl), ~m['ov']
        out['scores'][f'{f}_{t}'] = {'kappa_all': cohen_kappa_score(a, b), 'kappa_wo': cohen_kappa_score(a[keep], b[keep]),
                                     'mf1_all': f1_score(a, b, average='macro'), 'mf1_wo': f1_score(a[keep], b[keep], average='macro'),
                                     'n_wo': int(keep.sum())}
import os
os.makedirs('data/analysis/final', exist_ok=True)
json.dump(out, open('data/analysis/final/fewshot_overlap.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
