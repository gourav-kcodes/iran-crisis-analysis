# Text of the three analysis reports (best | zs | fs). Every number is read from
# data/analysis/report_<s>/results.json; every ranking claim ("highest of the eight regions") is checked here.
import json
import numpy as np

RES = {s: json.load(open(f'data/analysis/report_{s}/results.json')) for s in ('best', 'zs', 'fs')}
WK = ['March 2–8', 'March 9–15', 'March 16–22', 'March 23–29', 'March 30 – April 5', 'April 6–12', 'April 13–19',
      'April 20–26', 'April 27 – May 3']
REG8 = ['US', 'GB', 'PK', 'NG', 'ID', 'AE', 'FR', 'DE']
LANG6 = ['English', 'Indonesian', 'German', 'Spanish', 'French', 'Catalan']
REGNAME = {'US': 'United States', 'GB': 'United Kingdom', 'PK': 'Pakistan', 'NG': 'Nigeria', 'ID': 'Indonesia',
           'AE': 'UAE', 'FR': 'France', 'DE': 'Germany', 'MY': 'Malaysia'}
SENT9 = ['Fear', 'Anger', 'Sadness', 'Hope', 'Solidarity', 'Frustration', 'Humour', 'Disgust', 'Curiosity']
FR10 = ['Political blame', 'Military/strategic', 'Economic/sanctions', 'Humanitarian', 'Religious framing', 'Satire/meme',
        'Historical context', 'Media criticism', 'Personal connection', 'Misinformation']
FR11 = FR10 + ['Unclear']


def A(cond, msg='claim check failed'):
    if not cond:
        raise AssertionError(msg)


def f1(x):
    return f'{x:.1f}'


def pc(x):
    return f'{x:.1f}%'


def ci(v):
    return f'{v[0]:.1f}% [{v[1]:.1f}–{v[2]:.1f}%]'


def cir(v):
    return f'[{v[1]:.1f}–{v[2]:.1f}%]'


def fp(p):
    if p < 0.001:
        return 'p < 0.001'
    if p < 0.1:
        return f'p = {p:.3f}'
    return f'p = {p:.2f}'


def pv(p):
    return '< 0.001' if p < 0.001 else (f'{p:.3f}' if p < 0.1 else f'{p:.2f}')


def nr(x):
    return f'{x:,.1f}'.rstrip('0').rstrip('.') if x < 1000 else f'{x:,.0f}'


def lst(items):
    items = list(items)
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def times(a, b):
    w = {2: 'twice', 3: 'three times', 4: 'four times', 5: 'five times', 6: 'six times'}
    return 'about ' + w[round(a / b)]


def num(x):
    return f'{x:g}' if x < 1000 else f'{x:,.0f}'


class Ctx:
    def __init__(self, S):
        self.S, self.R = S, RES[S]
        R = self.R
        self.ss = 'zero-shot' if S in ('best', 'zs') else 'few-shot'      # Mistral sentiment setting
        self.fs_ = 'zero-shot' if S == 'zs' else 'few-shot'              # Mistral frame setting
        self.qs = 'zero-shot' if S == 'zs' else 'few-shot'               # Qwen setting
        self.wk = list(R['week_totals'])
        self.E = json.load(open(f'data/analysis/report_{S}/classification_error.json'))

    def ec(self, name):
        return self.E['compare'][name]

    def prec(self, lab):
        v = self.E['precision'][lab]
        return v['k'], v['n']

    # shares
    def d(self, model, kind, lab):
        return self.R['dist'][model][kind].get(lab, 0.0)

    def sh(self, nm, kind, g, lab, m='m'):
        return self.R[f'{nm}_{kind}_{m}'][g].get(lab, 0.0)

    def groups(self, nm):
        return REG8 if nm == 'reg' else LANG6

    def rank(self, nm, kind, lab, g, m='m'):
        v = sorted(self.groups(nm), key=lambda x: -self.sh(nm, kind, x, lab, m))
        return v.index(g) + 1

    def top(self, nm, kind, lab, m='m'):
        return max(self.groups(nm), key=lambda x: self.sh(nm, kind, x, lab, m))

    def hi(self, nm, kind, g, lab, m='m', r=1):
        A(self.rank(nm, kind, lab, g, m) == r, f'{g} not rank {r} for {lab} ({self.S})')
        return pc(self.sh(nm, kind, g, lab, m))

    def W(self, tag, lab):
        return [self.R['weekly'][tag].get(lab, {}).get(w, 0.0) for w in self.wk]

    def peak(self, tag, lab):
        w = self.W(tag, lab)
        return WK[w.index(max(w))], max(w)

    def tt(self, key):
        return self.R['trend_tests'][key]

    def pt(self, key):
        return self.R['pair_tests'][key]

    def co(self, s, f, m='m'):
        return self.R[f'cooc_{m}'][s][f]

    def cotop(self, s, m='m', k=1, skip_unclear=True):
        row = self.R[f'cooc_{m}'][s]
        fr = sorted([f for f in FR11 if not (skip_unclear and f == 'Unclear')], key=lambda f: -row[f]['pct'])
        return fr[:k] if k > 1 else fr[0]

    def es(self, lab):
        return self.R['eng_sent_m'][lab]

    def ef(self, lab):
        return self.R['eng_frame_m'][lab]

    def pcl(self, model, tier, lab):
        return self.R['perclass'][model][tier][lab]


# ─────────────────────────────────────────────────────────────────────────────
def build(d, S):
    c = Ctx(S)
    R = c.R
    ss, fs_, qs = c.ss, c.fs_, c.qs
    SZ, FZ = ss == 'zero-shot', fs_ == 'zero-shot'
    M = f'Mistral 7B'
    msent, mframe = f'Mistral 7B {ss}', f'Mistral 7B {fs_}'
    qn = f'Qwen2.5-3B {qs}'
    coll = R['collection']
    ev = R['eval']

    title = {'best': 'Substantive Analysis Report', 'zs': 'Strict Zero-Shot Analysis Report',
             'fs': 'Strict Few-Shot Analysis Report'}[S]
    d.title(title, 'Iran Crisis — Public Sentiment & Narrative Framing on TikTok  |  March–April 2026')

    # ── numbers used in several places ────────────────────────────────────────
    gfc = R['gold_frame_ci']
    fear_s, anger_s = R['ci_sent']['Fear'], R['ci_sent']['Anger']
    emo_rank = sorted(SENT9, key=lambda l: -c.d('Mistral', 'sent', l))
    pb, ec, mil = R['ci_frame']['Political blame'], R['ci_frame']['Economic/sanctions'], R['ci_frame']['Military/strategic']
    top_frame = max(FR10, key=lambda l: c.d('Mistral', 'frame', l))
    ap = c.co('Anger', 'Political blame')
    ap_ci = R['cooc_ci']['Anger']['Political blame']
    sol_t, hum_t = c.tt('Solidarity_m'), c.tt('Humanitarian_m')
    id_rel = c.pt('Indonesian|English|Religious framing')
    gb_sol = c.pt('GB|US|Solidarity')
    nm_e = 100 - R['gold_counts']['t1']['Neutral'] / 4

    # ── Summary ───────────────────────────────────────────────────────────────
    E = c.E
    osent, ofr = E['overall_sentiment'], E['overall_frame']
    rel_c, solw_c = c.ec('Religious framing: Indonesian vs English'), c.ec('Solidarity: first two weeks vs last two')
    pbw_c, huw_c = c.ec('Political blame: first two weeks vs last two'), c.ec('Humanitarian: first two weeks vs last two')
    gbs_c, ngr_c = c.ec('Solidarity: GB vs US'), c.ec('Religious framing: NG vs US')
    hci_c = c.ec('Historical context: Indonesian vs English')
    A(rel_c['p'] < 0.05 and ngr_c['p'] < 0.05 and pbw_c['p'] < 0.05 and huw_c['p'] < 0.05)
    A(solw_c['p'] > 0.05 and gbs_c['p'] > 0.05 and hci_c['p'] > 0.05)
    fear_up = osent['Fear']['gold_exact'][1]
    emo_max = max(SENT9, key=lambda l: osent[l]['corrected'])
    A(emo_rank.index('Fear') <= 1)
    d.h1('Summary')
    setting_line = {
        'best': 'Labels come from the best-performing setting for each tier: Mistral 7B zero-shot for sentiment and '
                'Mistral 7B few-shot for frames, with Qwen2.5-3B few-shot as the cross-model check.',
        'zs': 'Every label comes from one setting, zero-shot, for both tiers: Mistral 7B as the primary model and '
              'Qwen2.5-3B zero-shot as the cross-model check.',
        'fs': 'Every label comes from one setting, few-shot, for both tiers: Mistral 7B as the primary model and '
              'Qwen2.5-3B few-shot as the cross-model check.'}[S]
    d.para(f'This report analyses 5,848 TikTok video descriptions about the Iran crisis (1 March – 30 April 2026). '
           f'{setting_line} Classification error is carried into the main estimates using the 400-row gold standard '
           f'(Sections 1.5 and 8.2). The main results:')
    d.bullet(f'Most posts carry no clear emotion. Annotators coded {R["gold_neutral_ci"][0]:.1f}% of a random sample of '
             f'400 videos as Neutral, while {msent} labels only {c.d("Mistral", "sent", "Neutral"):.1f}% Neutral. Once '
             f'classifier error is included, no emotional label reaches {np.ceil(osent[emo_max]["ci"][1]):.0f}% of videos. '
             f'Fear, {"the most frequent" if emo_rank[0] == "Fear" else "the second most frequent"} emotional label in Mistral\'s '
             f'output ({pc(c.d("Mistral", "sent", "Fear"))}), never appears in the gold sample (true share below {fear_up:.1f}%, 95% upper bound). Sentiment results in this '
             f'report therefore describe the classifier\'s labels, not confirmed content.')
    d.bullet(f'Corrected for classifier error, the frame mix is Military/strategic {ci([ofr["Military/strategic"]["corrected"]] + ofr["Military/strategic"]["ci"])}, '
             f'Economic/sanctions {ci([ofr["Economic/sanctions"]["corrected"]] + ofr["Economic/sanctions"]["ci"])} and '
             f'Political blame {ci([ofr["Political blame"]["corrected"]] + ofr["Political blame"]["ci"])}, matching the human-coded '
             f'sample. Mistral\'s raw output puts {top_frame} first ({pc(c.d("Mistral", "frame", top_frame))}) because it '
             f'over-assigns Political blame.')
    d.bullet('The weekly video counts are an artefact of how the data were collected (two-day query windows plus '
             'month-end bulk queries), not of posting activity. Trends are therefore reported as weekly shares.')
    d.bullet(f'Findings that hold after classifier error: Indonesian-language posts use Religious framing more than '
             f'English posts (corrected {pc(rel_c["corr_a"])} vs {pc(rel_c["corr_b"])}, {fp(rel_c["p"])}); Nigerian posts '
             f'also exceed US posts ({fp(ngr_c["p"])}); Political blame rises from early March to late April and '
             f'Humanitarian framing falls ({fp(pbw_c["p"])} and {fp(huw_c["p"])}).')
    d.bullet(f'Findings that are significant in the classifier output but not confirmed once its error is included: the '
             f'Solidarity decline over time ({fp(solw_c["p"])}), higher Solidarity in British than US posts '
             f'({fp(gbs_c["p"])}), and higher Historical context in Indonesian'
             + (' and French' if c.pt('French|English|Historical context')['p'] < 0.05 else '')
             + f' posts. The gold sample has too '
             f'few rows of these labels to confirm or reject them.')
    k_ec, n_ec = c.prec('Economic/sanctions')
    sl, slv = c.es('Solidarity'), c.ef('Economic/sanctions')
    d.bullet(f'Economic/sanctions posts, the most reliably classified frame ({k_ec} of {n_ec} predictions confirmed on the '
             f'gold sample), receive the fewest likes (median {num(slv["med_likes"])}, {fp(slv["mw_likes_p"])}). Posts the '
             f'classifier labels Solidarity receive the most likes (median {num(sl["med_likes"])}), but only '
             f'{c.prec("Solidarity")[0]} of {c.prec("Solidarity")[1]} Solidarity predictions '
             f'{"was" if c.prec("Solidarity")[0] == 1 else "were"} confirmed.')
    A(sl['mw_likes_p'] < 0.001 and slv['mw_likes_p'] < 0.001)

    # ── 1. Dataset & model settings ──────────────────────────────────────────
    d.h1('1. Dataset & Model Settings')
    intro = {
        'best': 'This report presents the substantive analysis of 5,848 TikTok video descriptions collected via the TikTok '
                'Research API covering March 1 – April 30, 2026. The analysis uses the best-performing classification '
                'setting per model per tier, as determined by the evaluation phase against the human gold standard (400 '
                'annotated rows).',
        'zs': 'This report presents the substantive analysis of 5,848 TikTok video descriptions (March 1 – April 30, 2026) '
              'using one classification setting throughout: zero-shot for both tiers. The best-setting report combined '
              'Mistral zero-shot sentiment with few-shot frames; here every label comes from the same zero-shot run. '
              'Sentiment-only results are therefore identical to the best-setting report, while frame results and the '
              'sentiment × frame co-occurrence change.',
        'fs': 'This report presents the substantive analysis of 5,848 TikTok video descriptions (March 1 – April 30, 2026) '
              'using one classification setting throughout: few-shot for both tiers. The best-setting report combined '
              'Mistral zero-shot sentiment with few-shot frames; here every label comes from the same few-shot run. '
              'Frame-only results are therefore identical to the best-setting report, while sentiment results and the '
              'sentiment × frame co-occurrence change.'}[S]
    d.para(intro)
    T = lambda m, t, k: f'{ev[m][t][k]:.3f}' if k != 'agree' else f'{ev[m][t][k]:.1f}%'
    ds_t1 = 'zero-shot' if S in ('best', 'zs') else 'few-shot'
    ds_t3 = 'zero-shot' if S == 'zs' else 'few-shot'
    rows = [['Mistral 7B Instruct', 'Tier 1', ss], ['Mistral 7B Instruct', 'Tier 3', fs_],
            ['Qwen2.5-3B-Instruct', 'Tier 1', qs], ['Qwen2.5-3B-Instruct', 'Tier 3', qs],
            ['DeepSeek-R1 ⚠', 'Tier 1', ds_t1], ['DeepSeek-R1 ⚠', 'Tier 3', ds_t3]]
    if S != 'fs':
        rows.append(['Twitter-RoBERTa', 'Tier 1 (3-class)', 'zero-shot'])
    trow = []
    for nm, tier, st in rows:
        m = {'Mistral': 'Mistral', 'Qwen2.5-3B-Instruct': 'Qwen', 'DeepSeek-R1': 'DeepSeek', 'Twitter-RoBERTa': 'RoBERTa'}[nm.split()[0]]
        t = 't1' if tier.startswith('Tier 1') else 't3'
        trow.append([nm, tier, st.capitalize(), T(m, t, 'kappa'), T(m, t, 'wf1'), T(m, t, 'mf1'), T(m, t, 'agree')])
    d.table(['Model', 'Tier', 'Setting', 'Kappa', 'Weighted F1', 'Macro F1', 'Agreement'], trow,
            [2150, 1350, 1100, 850, 1150, 1000, 1250], size=19)
    nds = R['n_deepseek_sent'], R['n_deepseek_frame']
    dsn = (f'{nds[0]:,} rows (zero-shot, {ev["DeepSeek"]["n_t1"]} gold rows) and {nds[1]:,} rows (few-shot, '
           f'{ev["DeepSeek"]["n_t3"]} gold rows)' if S == 'best' else
           f'{nds[0]:,} of 5,848 videos ({ev["DeepSeek"]["n_t1"]} of 400 gold rows)')
    d.para(f'⚠ DeepSeek classification is partial — {dsn}. Results are indicative only.'
           + (' Twitter-RoBERTa has no few-shot setting and is not included.' if S == 'fs' else ''), italic=True)
    d.para('Two annotators coded the 400 gold rows independently. Their agreement was κ = 0.85 for Tier 1 and κ = 0.89 for '
           'Tier 3, so the gold labels themselves are reliable. Macro F1 weights every label equally and is the better '
           'guide to how well the minority labels are classified; weighted F1 is dominated by the Neutral rows.')
    if S == 'zs':
        d.para(f'Qwen\'s Tier 1 Kappa ({ev["Qwen"]["t1"]["kappa"]:.3f}) is close to chance, so its sentiment comparisons '
               f'carry little weight.')
    if S == 'fs':
        d.para(f'In this setting Mistral has the lowest Tier 1 Kappa of the three LLMs ({ev["Mistral"]["t1"]["kappa"]:.3f}), '
               f'while its Tier 3 Kappa ({ev["Mistral"]["t3"]["kappa"]:.3f}) is moderate.')

    # 1.1 data and methods
    d.h2('1.1  Data and Methods')
    d.bullet(f'5,848 videos from the TikTok Research API, retrieved with 18 Iran-related hashtags (the largest are '
             f'#OilPrices, #IranConflict, #StraitOfHormuz, #IranWar and #Iran). Each video appears once.',
             'Data: ')
    d.bullet(f'Nine two-day query windows, each a Sunday and Monday from 1–2 March to 26–27 April '
             f'({coll["window_n"]:,} videos), plus one query for each whole month ({coll["month_query_n"]:,} videos). The '
             f'month queries returned mostly videos from the last days of each month. Section 3.1 shows what this means '
             f'for the timeline.', 'Collection: ')
    d.bullet('Each post was classified from the English translation of its description. Speech transcripts exist for '
             '949 videos but were not used.', 'Input text: ')
    d.bullet('Sentiment and frame were classified in two separate prompts. In the few-shot runs, descriptions shorter '
             f'than 10 characters or 4 words were set to Unclear on both tiers without calling the model '
             f'({R["short"]["n"]} videos; {R["short"]["ne_share"]:.1f}% of non-English posts against '
             f'{R["short"]["en_share"]:.1f}% of English posts). The zero-shot runs sent every post to the model.',
             'Prompts: ')
    d.bullet(f'Language is the detected language of the description; region is the region code reported by the API. '
             f'Communities with 50 or more videos are compared, and the six largest languages and eight largest regions '
             f'are shown in the figures.', 'Language and region: ')
    d.bullet(f'Weeks run Monday to Sunday; the nine weeks with at least 30 videos (March 2 – May 3) are used. '
             f'{R["n_no_date"]} videos have no timestamp and are left out of the time analysis. The final week only has '
             f'data for April 27–30.', 'Weeks: ')
    d.bullet('95% bootstrap confidence intervals (2,000 resamples) for shares and medians. Chi-square tests with '
             'Cramér\'s V for label differences across languages and regions, Fisher\'s exact test for pairwise '
             'comparisons and co-occurrence, the Cochran–Armitage test for weekly trends, and Mann–Whitney U and '
             'Kruskal–Wallis tests for engagement. Many tests are run, so p-values between 0.01 and 0.05 should be read '
             'with caution.', 'Uncertainty and tests: ')

    # 1.2 reliability
    d.h2('1.2  Classification Reliability — Per-Class Scores')
    n_neu = R['gold_counts']['t1']['Neutral']
    d.para(f'Weighted F1 is high mainly because {n_neu} of the 400 gold rows are Neutral, so it says little about the '
           f'emotional labels. The charts below show per-class Precision, Recall, and F1 for every label, with the number '
           f'of gold rows and predictions under each label. Emotional labels have between 0 and 10 gold rows each, so '
           f'their scores are rough indications. Differences of 1–3 percentage points in downstream distributions should '
           f'be treated with caution where per-class F1 is below 0.40.')
    d.fig(1, f'Per-class Precision, Recall, and F1 for Tier 1 sentiment labels — {msent}. Gold-row and prediction counts '
             f'shown under each label; Fear and Sadness have no gold rows.')
    d.fig(2, f'Per-class Precision, Recall, and F1 for Tier 3 frame labels — {mframe}.')
    P1 = lambda l: c.pcl('Mistral', 't1', l)
    P3 = lambda l: c.pcl('Mistral', 't3', l)
    neu = P1('Neutral')
    pe = R['pred_emotional_gold']
    d.bullet(f'Precision {neu["p"]:.3f}, recall {neu["r"]:.3f}. Almost every post Mistral calls Neutral is Neutral, but '
             f'it gives an emotional label to {pe:.1f}% of gold rows, against {nm_e:.1f}% for the annotators. '
             + ('Emotional-label shares in this report are therefore upper bounds.' if SZ else
                'Emotional-label percentages in this report are upper-bound estimates; read them as relative patterns, '
                'not levels.'),
             'Neutral: ')
    fear_n = P1('Fear')['npred']
    A(R['fear_pred_gold_labels'] == {'Neutral': fear_n})
    d.bullet(f'The gold standard contains no Fear rows. Mistral predicted Fear for {fear_n} of the 400 gold rows, and the '
             f'annotators coded all {fear_n} as Neutral, so Fear precision on the sample is 0 and recall cannot be '
             f'estimated. With 0 Fear rows in a random sample of 400, the true Fear share is below '
             f'{c.E["overall_sentiment"]["Fear"]["gold_exact"][1]:.1f}% (exact 95% upper bound). Fear findings in this '
             f'report describe what the classifier labels Fear, not confirmed fear in the posts.', 'Fear (no gold rows): ')
    if not SZ:
        over = sorted([l for l in SENT9 if P1(l)['npred'] >= 4 * max(1, P1(l)['support']) and P1(l)['support'] > 0],
                      key=lambda l: -P1(l)['npred'])
        d.bullet('; '.join(f'{l} {P1(l)["npred"]} predictions against {P1(l)["support"]} gold '
                           f'row{"s" if P1(l)["support"] > 1 else ""}' for l in over) + '.',
                 'Over-assigned emotional labels: ')
    else:
        d.bullet(f'Anger was predicted {P1("Anger")["npred"]} times against {P1("Anger")["support"]} gold rows, '
                 f'Curiosity {P1("Curiosity")["npred"]} times against {P1("Curiosity")["support"]}.',
                 'Anger and Curiosity (over-assigned): ')
    ms, pbm = P3('Military/strategic'), P3('Political blame')
    d.bullet(f'{ms["npred"]} predictions against {ms["support"]} gold rows. The classifier under-detects this frame, so '
             f'findings about it are lower-bound estimates.',
             f'Military/strategic (F1={ms["f1"]:.3f}, recall={ms["r"]:.3f}): ')
    d.bullet(f'{pbm["npred"]} predictions against {pbm["support"]} gold rows, and {R["mil_as_pb"]} of the '
             f'{ms["support"]} gold Military/strategic rows were labelled Political blame. The classifier over-assigns '
             f'this label. Counts are upper-bound estimates.',
             f'Political blame (F1={pbm["f1"]:.3f}, precision={pbm["p"]:.3f}): ')
    ecm, hu = P3('Economic/sanctions'), P1('Humour')
    A(ecm['f1'] == max(P3(l)['f1'] for l in FR10))
    d.bullet(f'Economic/sanctions (F1={ecm["f1"]:.3f}) and Humour (F1={hu["f1"]:.3f}, {hu["support"]} gold rows) are the '
             f'most reliably classified labels — findings about them carry the most confidence.', 'Most reliable: ')
    d.bullet(f'Unclear frame F1 is {P3("Unclear")["f1"]:.3f} ({P3("Unclear")["npred"]} predictions, '
             f'{P3("Unclear")["support"]} gold rows). Misinformation has no gold rows and '
             f'{P3("Misinformation")["npred"]} prediction{"s" if P3("Misinformation")["npred"] != 1 else ""} on the sample, so it cannot be checked.', 'Unclear and Misinformation: ')
    d.para(f'The table gives the same per-class F1 for the cross-check model, {qn}, next to Mistral. Qwen classifies '
           f'Neutral with F1 {c.pcl("Qwen", "t1", "Neutral")["f1"]:.3f} against Mistral\'s {neu["f1"]:.3f}, and its '
           f'emotional labels are no more reliable, so agreement between the two models on an emotional label does not '
           f'show that the label is correct.')
    qrows = []
    for t, labs in (('t1', SENT9 + ['Neutral']), ('t3', FR11)):
        for l in labs:
            a, b = c.pcl('Mistral', t, l), c.pcl('Qwen', t, l)
            qrows.append([l, str(a['support']), str(a['npred']), f'{a["f1"]:.2f}', str(b['npred']), f'{b["f1"]:.2f}'])
    d.table(['Label', 'Gold rows', 'Mistral pred.', 'Mistral F1', 'Qwen pred.', 'Qwen F1'], qrows,
            [2600, 1200, 1300, 1200, 1200, 1200], size=18)
    d.caption('Table 1. Per-class F1 on the 400 gold rows — Mistral 7B and Qwen2.5-3B. Upper block: Tier 1 sentiment; '
              'lower block: Tier 3 frame.')
    d.fig(3, f'Frame mix on the 400 gold rows: annotators (with 95% bootstrap CIs) against {mframe} and {qn}.')
    A(gfc['Military/strategic'][0] == max(v[0] for v in gfc.values()))
    gm_pb = 100 * pbm['npred'] / 400
    gm_ms = 100 * ms['npred'] / 400
    d.para(f'Because the gold standard is a simple random sample of the same videos, it gives a direct check on the '
           f'frame mix. Annotators coded Military/strategic in {ci(gfc["Military/strategic"])} of gold rows, '
           f'Economic/sanctions in {ci(gfc["Economic/sanctions"])} and Political blame in {ci(gfc["Political blame"])}. '
           f'On the same 400 videos Mistral assigns Political blame to {gm_pb:.1f}% and Military/strategic to '
           f'{gm_ms:.1f}%. The classifier\'s frame shares therefore overstate Political blame and understate '
           f'Military/strategic; the gold sample is the better estimate of the true frame mix.')

    # 1.3 CIs
    d.h2('1.3  Bootstrap Confidence Intervals — Key Label Proportions')
    d.para('To quantify uncertainty, 95% bootstrap confidence intervals (2,000 resamples) were computed for all key label '
           'proportions. These intervals reflect sampling variation in the classification output, not classifier error.')
    d.fig(4, f'Emotional sentiment label proportions with 95% bootstrap confidence intervals — {msent} (n=5,848). '
             f'Neutral excluded.')
    d.fig(5, f'Frame label proportions with 95% bootstrap confidence intervals — {mframe} (n=5,848). Unclear excluded.')
    keys = ', '.join(f'{l} {ci(R["ci_sent"][l])}' for l in emo_rank[:4 if not SZ else 2])
    d.para(f'Key CI observations: {keys}; Political blame {ci(pb)}; Economic/sanctions {ci(ec)}; Military/strategic '
           f'{ci(mil)}. The intervals are narrow for the full dataset, so sampling variation is small; classification '
           f'error (Section 1.2) is the larger source of uncertainty. Comparisons between communities carry wider '
           f'intervals and are tested separately.')

    # 1.4 non-English
    d.h2('1.4  Non-English Content')
    ng = R['ne_gold']
    neo = R['ne_overall']
    d.para('Every post was classified from its English translation, so non-English posts depend on the translation and '
           'are more often short or hashtag-only. English-language and non-English-language posts were compared within '
           'each region to test whether high Unclear and Neutral rates reflect the posts themselves or classification '
           'problems. Region codes used throughout: US = United States, GB = United Kingdom, PK = Pakistan, NG = '
           'Nigeria, ID = Indonesia, AE = United Arab Emirates, FR = France, DE = Germany.')
    ne = R['nonenglish']
    d.table(['Region', 'Unclear frame (All)', 'Unclear (English)', 'Unclear (Non-English)', 'Neutral (English)',
             'Neutral (Non-English)', 'Non-English n'],
            [[g, pc(ne[g]['unc_all']), pc(ne[g]['unc_en']), pc(ne[g]['unc_ne']), pc(ne[g]['neu_en']), pc(ne[g]['neu_ne']),
              str(ne[g]['n_ne'])] for g in REG8], [900, 1450, 1300, 1450, 1300, 1450, 1100], size=18)
    d.caption(f'Table 2. Unclear frame and Neutral sentiment rates for English and non-English posts by region — '
              f'Mistral 7B ({fs_} frame, {ss} sentiment).')
    exc = [g for g in REG8 if ne[g]['unc_ne'] <= ne[g]['unc_en']]
    A(exc == ['ID'])
    d.para(f'Non-English posts are coded Unclear far more often in every region except Indonesia, which has only '
           f'{ne["ID"]["n_en"]} English posts. The gold standard indicates that most of this gap comes from the posts, not '
           f'the classifier. Annotators coded {ng["gold_unc_ne"]:.1f}% of the {ng["n_ne"]} non-English gold rows as '
           f'Unclear, against {ng["gold_unc_en"]:.1f}% of the {ng["n_en"]} English rows. Mistral predicted Unclear for '
           f'{ng["pred_unc_ne"]:.1f}% of the non-English gold rows, and {ng["prec_unc_ne"]:.1f}% of those predictions '
           f'matched the annotators: many non-English posts are short or consist mainly of hashtags, and human coders '
           f'could not assign them a frame either. Where annotators did assign a specific frame to a non-English post '
           f'({ng["n_framed_ne"]} rows), Mistral matched them in {ng["agree_framed_ne"]:.1f}% of cases, against '
           f'{ng["agree_framed_en"]:.1f}% for English posts ({ng["n_framed_en"]} rows), so frame assignment on non-English '
           f'text is {"somewhat " if ng["agree_framed_en"] - ng["agree_framed_ne"] < 10 else ""}weaker, but this rests on '
           f'few rows.')
    d.para(f'Neutral shows the same pattern. Across the full data, {ci(neo["neu_ne"])} of non-English posts are Neutral '
           f'against {ci(neo["neu_en"])} of English posts. In the gold sample the annotators\' gap is small '
           f'({ng["gold_neu_ne"]:.1f}% vs {ng["gold_neu_en"]:.1f}%), while Mistral\'s is larger ({ng["pred_neu_ne"]:.1f}% '
           f'vs {ng["pred_neu_en"]:.1f}%). Non-English posts are slightly more often neutral in fact, and the classifier '
           f'adds to the gap — most likely because translated, short posts give it fewer emotional cues. Several regional '
           f'non-English percentages also rest on small counts ({ne["AE"]["n_ne"]} posts in AE, {ne["NG"]["n_ne"]} in NG).')
    d.para('Two limits apply to this check. First, the annotation guideline told annotators to read the English '
           'translation first and the original only when the translation looked poor. Agreement on Unclear therefore '
           'shows that the problem is not specific to Mistral, but it cannot fully separate a post that has no frame '
           'from a translation that lost it. Second, the gold sample has few non-English rows: 17 Indonesian, 17 German '
           '(15 of them coded Unclear by the annotators), 13 French and 11 Spanish. Findings for non-English communities '
           'are therefore reported as differences in the classifier output, without interpretation of the communities '
           'themselves.')

    # 1.5 classification error
    d.h2('1.5  Carrying Classification Error into the Estimates')
    osent, ofr = c.E['overall_sentiment'], c.E['overall_frame']
    d.para('The confidence intervals above cover sampling variation only. To carry classification error into the '
           'estimates, the gold standard is used as a calibration sample. Because it is a random sample of the same '
           'videos, it shows, for every label Mistral predicts, how often the annotators gave each true label. An '
           'error-corrected share is the classifier\'s predicted mix multiplied by these rates. The bootstrap (2,000 '
           'draws) resamples both the gold rows and the videos, so the interval includes classifier error as well as '
           'sampling variation. The method assumes the classifier makes the same kinds of errors in every group and week.')
    rows = []
    for tier, labs, o in (('Frame', FR11, ofr), ('Sentiment', SENT9 + ['Neutral'], osent)):
        for l in labs:
            v = o[l]
            corr = (f'{v["corrected"]:.1f}% [{v["ci"][0]:.1f}–{v["ci"][1]:.1f}%]' if v['gold_n'] > 0 else 'not estimable')
            rows.append([l, tier, pc(v['raw']), corr,
                         f'{v["gold"]:.1f}% [{v["gold_exact"][0]:.1f}–{v["gold_exact"][1]:.1f}%]', str(v['gold_n'])])
    d.table(['Label', 'Tier', 'Classifier', 'Error-corrected [95% CI]', 'Gold sample [exact 95% CI]', 'Gold rows'], rows,
            [2300, 1100, 1100, 2100, 2100, 900], size=18)
    d.caption(f'Table 3. Classifier shares, error-corrected shares and gold-sample shares — {mframe} frames and {msent} '
              f'sentiment. "Not estimable": the label has no gold rows.')
    pbo, mso = ofr['Political blame'], ofr['Military/strategic']
    hum_o = osent['Humour']
    d.para(f'For frames, the correction moves the classifier\'s mix close to the human-coded sample: Political blame falls '
           f'from {pc(pbo["raw"])} to {pc(pbo["corrected"])} and Military/strategic rises from {pc(mso["raw"])} to '
           f'{pc(mso["corrected"])}. For sentiment, the corrected shares of emotional labels are all small — Humour, the '
           f'best-supported, is {hum_o["corrected"]:.1f}% [{hum_o["ci"][0]:.1f}–{hum_o["ci"][1]:.1f}%] — and Fear and '
           f'Sadness cannot be estimated because the gold sample contains none. The group comparisons and trends in '
           f'Sections 3–7 use the classifier\'s labels with sampling intervals; Section 8.2 re-tests the main ones with '
           f'classification error included.')

    # ── 2. Overall distributions ─────────────────────────────────────────────
    d.h1('2. Overall Distributions')
    mn, qnn, dn = c.d('Mistral', 'sent', 'Neutral'), c.d('Qwen', 'sent', 'Neutral'), c.d('DeepSeek', 'sent', 'Neutral')
    rbn = R['roberta_dist']['Neutral']
    qemo = sorted(SENT9, key=lambda l: -c.d('Qwen', 'sent', l))
    if S == 'best':
        d.para(f'Neutral sentiment dominates across all models — ranging from {dn:.0f}% (DeepSeek zero-shot) to {mn:.0f}% '
               f'(Mistral zero-shot). This reflects the largely informational nature of TikTok video descriptions during '
               f'the crisis, where creators describe events without explicitly expressing personal emotion; annotators '
               f'coded {R["gold_neutral_ci"][0]:.1f}% of the gold rows as Neutral, so all models assign emotional labels '
               f'more often than the human coding supports. Fear is the most frequent emotional label for all three LLMs '
               f'(Mistral {ci(fear_s)}). Anger is second for Mistral ({ci(anger_s)}) and DeepSeek, while Qwen few-shot '
               f'assigns Humour ({pc(c.d("Qwen", "sent", "Humour"))}) more often than Anger '
               f'({pc(c.d("Qwen", "sent", "Anger"))}).')
        A(qemo[0] == 'Fear' and qemo[1] == 'Humour' and emo_rank[:2] == ['Fear', 'Anger'])
    elif S == 'zs':
        d.para(f'Neutral dominates sentiment for Mistral ({pc(mn)}) and DeepSeek ({pc(dn)}), and RoBERTa labels '
               f'{pc(rbn)} of videos Neutral. The annotators coded {R["gold_neutral_ci"][0]:.1f}% of gold rows as Neutral, '
               f'so all models assign emotional labels more often than the human coding supports. Among emotional labels, '
               f'Fear ({pc(fear_s[0])}) and Anger ({pc(anger_s[0])}) are the most frequent for Mistral. Qwen zero-shot '
               f'assigns Fear to {pc(c.d("Qwen", "sent", "Fear"))} of videos — more than Neutral ({pc(qnn)}) — which '
               f'reflects its near-chance Tier 1 agreement rather than the content.')
        A(c.d('Qwen', 'sent', 'Fear') > qnn)
    else:
        d.para(f'Few-shot prompting moves Mistral far from Neutral: {pc(mn)} of videos are Neutral, against {pc(qnn)} for '
               f'Qwen few-shot, {pc(dn)} for DeepSeek few-shot and {R["gold_neutral_ci"][0]:.1f}% in the gold sample. The '
               f'leading emotional labels for Mistral are '
               + lst(f'{l} ({pc(c.d("Mistral", "sent", l))})' for l in emo_rank[:4])
               + f' — three of which are heavily over-predicted on the gold sample. Qwen few-shot puts Fear first '
                 f'({pc(c.d("Qwen", "sent", "Fear"))}), followed by {qemo[1]} ({pc(c.d("Qwen", "sent", qemo[1]))}) and '
                 f'{qemo[2]} ({pc(c.d("Qwen", "sent", qemo[2]))}).')
        A(qemo[0] == 'Fear')
    top3 = sorted(FR10, key=lambda l: -c.d('Mistral', 'frame', l))[:3]
    s3 = sum(c.d('Mistral', 'frame', l) for l in top3)
    qtop = max(FR10, key=lambda l: c.d('Qwen', 'frame', l))
    dtop = max(FR10, key=lambda l: c.d('DeepSeek', 'frame', l))
    d.para(f'For frames, ' + lst(f'{l} ({ci(R["ci_frame"][l])})' for l in top3)
           + f' together account for {s3:.1f}% of Mistral\'s classified videos. '
           + (f'Political blame and Economic/sanctions are within half a percentage point and their intervals overlap, so '
              f'neither is clearly dominant in Mistral\'s output. ' if not FZ else '')
           + f'Qwen {qs} puts {qtop} first ({pc(c.d("Qwen", "frame", qtop))}) and DeepSeek puts {dtop} first '
             f'({pc(c.d("DeepSeek", "frame", dtop))}); the human-coded gold sample has Military/strategic first '
             f'({pc(gfc["Military/strategic"][0])}).'
           + (f' Qwen also assigns Misinformation to {pc(c.d("Qwen", "frame", "Misinformation"))} of videos, a label with '
              f'no gold rows.' if S == 'zs' else ''))
    if not FZ:
        A(abs(pb[0] - ec[0]) < 0.6 and pb[1] < ec[2])
    d.para('The variation in Neutral rates across models reflects classification sensitivity differences documented in '
           'the evaluation phase, not genuine content differences.')
    d.fig(6, f'Sentiment distribution by model — {"best-performing setting per model" if S == "best" else ss + " setting for all models"}.'
          + (' RoBERTa is three-class, so only its Neutral share is shown.' if S != 'fs' else ''))
    d.fig(7, f'Frame distribution by model — {fs_} setting for all three LLMs.')

    # ── 3. Trends ────────────────────────────────────────────────────────────
    d.h1('3. Sentiment & Frame Trends Over Time')
    d.h2('3.1  Collection Design and Events')
    wt, wtn = R['week_totals'], R['week_totals_nomonth']
    br = coll['by_range']
    wins = [br[k] for k in sorted(br) if k[:8].isdigit() and k not in ('20260301_20260331', '20260401_20260430')]
    A(len(wins) == 9)
    daily = R['daily']
    d.para(f'Weekly video counts reflect how the data were collected, not how much was posted. Each two-day window '
           f'returned {min(wins)}–{max(wins)} videos, mostly from the Monday. The two month queries '
           f'({br["20260301_20260331"]} videos for March, {br["20260401_20260430"]} for April) returned mostly videos from '
           f'the last days of each month — {daily["2026-03-31"]["month"]} posted on March 31 and '
           f'{daily["2026-04-30"]["month"]} on April 30. This is why the weeks of March 30 – April 5 '
           f'({wt[c.wk[4]]:,} videos) and April 27 – May 3 ({wt[c.wk[8]]:,}) look like surges. Without the month-query '
           f'videos, weekly totals range from {min(wtn.values())} to {max(wtn.values())}, and the lowest is the final week, '
           f'which holds only the Monday of the last window. Volume is therefore not interpreted in this report: all trends are weekly shares, and each '
           f'trend was re-tested without the month-query videos (Section 3.5).')
    d.para('The war began on February 28 with US and Israeli strikes on Iran, one day before the data start. Verified '
           'events inside the period, marked in Figure 8: on March 25 Iran rejected a US 15-point proposal passed on by '
           'Pakistan; on April 7 a two-week ceasefire was announced, after US threats of major strikes ahead of a '
           'deadline; on April 12 the Islamabad talks collapsed and the US announced a naval blockade starting April 13; '
           'on April 21 the ceasefire was extended indefinitely while the blockade stayed in place. Two sampling windows '
           'fall right on these events: April 5–6, just before the ceasefire, and April 12–13, the talks-collapse and '
           'blockade days. Because each week is mostly one Sunday–Monday window, a weekly share describes those two days, '
           'not the whole week.')
    d.fig(8, 'Videos per day by collection query, with verified events. Two-day window queries in blue, month queries in '
             'red.')

    # 3.2 sentiment trend
    d.h2('3.2  Sentiment Trend')
    d.fig(9, f'Sentiment trend — {msent}.')
    d.fig(10, f'Sentiment trend — {qn}.')
    Ws, Wq = lambda l: c.W('sent_m', l), lambda l: c.W('sent_q', l)
    sol, solq = Ws('Solidarity'), Wq('Solidarity')
    A(sol[0] == max(sol) and solq[0] == max(solq))
    ang = Ws('Anger')
    fpk, fpv = c.peak('sent_m', 'Fear')
    apk, apv = c.peak('sent_m', 'Anger')
    if SZ:
        A(apk == WK[5] and ang[0] == min(ang))
        d.para(f'Solidarity is highest in the first week ({pc(sol[0])} of weekly videos in Mistral) and declines steadily '
               f'to {pc(sol[-1])} by late April, and Qwen shows the same decline ({pc(solq[0])} to {pc(solq[-1])}). '
               f'This is the clearest sentiment trend in the classifier output, but it is not confirmed once '
               f'classification error is included (Section 8.2). Anger is lowest in the first week ({pc(ang[0])}) '
               f'and peaks in April 6–12 ({pc(apv)}); Fear also peaks in April 6–12 ({pc(fpv)}). That week is mostly the '
               f'April 6 window, the day before the ceasefire was announced. Hope stays low throughout '
               f'({min(Ws("Hope")):.1f}–{max(Ws("Hope")):.1f}% per week).')
        A(fpk == WK[5])
    else:
        fr_ = Ws('Frustration')
        A(apk == WK[5] and fpk == WK[5])
        other = [v for i, v in enumerate(ang) if i != 5]
        d.para(f'Solidarity is highest in the first week ({pc(sol[0])}) and declines to {pc(sol[-1])} by late April. Qwen '
               f'shows the same decline ({pc(solq[0])} to {pc(solq[-1])}). This is the clearest sentiment trend in the '
               f'classifier output, but it is not confirmed once classification error is included (Section 8.2). Anger nearly doubles in April 6–12 ({pc(apv)}), against {min(other):.1f}–'
               f'{max(other):.1f}% in every other week, and Fear also peaks that week ({pc(fpv)}). That week is mostly the '
               f'April 6 window, the day before the ceasefire was announced. Frustration jumps from {pc(fr_[0])} in the '
               f'first week to {min(fr_[1:]):.1f}–{max(fr_[1:]):.1f}% afterwards.')
    d.fig(11, f'Remaining sentiment labels — {msent}.')
    d.fig(12, f'Remaining sentiment labels — {qn}.')
    hum, cur, dis = Ws('Humour'), Ws('Curiosity'), Ws('Disgust')
    dpk, dpv = c.peak('sent_m', 'Disgust')
    if SZ:
        cpk, cpv = c.peak('sent_m', 'Curiosity')
        hq = Wq('Humour')
        A(dpk == WK[5])
        d.para(f'Humour rises from {pc(hum[0])} in the first week to {pc(hum[5])} in April 6–12 and ends at '
               f'{pc(hum[-1])} — a change of about one percentage point, small relative to weekly sample sizes; Qwen '
               f'{qs} shows no clear Humour trend ({min(hq):.1f}–{max(hq):.1f}% per week). Curiosity stays between '
               f'{min(cur):.1f}% and {max(cur):.1f}% and Disgust between {min(dis):.1f}% and {max(dis):.1f}%. '
               f'Weekly movements of these small labels are 1–3 percentage points on a few hundred videos, so they are not read as findings; Section 3.5 tests the trends.')
    else:
        A(dpk == WK[5])
        d.para(f'Curiosity stays between {min(cur):.1f}% and {max(cur):.1f}% every week — a level that reflects the '
               f'over-assignment shown in Section 1.2 rather than the posts. Humour moves between {min(hum):.1f}% and '
               f'{max(hum):.1f}% and Disgust between {min(dis):.1f}% and {max(dis):.1f}%. Weekly movements of these small labels are 1–3 percentage points on a few hundred videos, so they are not read as findings; Section 3.5 tests the trends.')

    # 3.3 frame trend
    d.h2('3.3  Frame Trend')
    d.fig(13, f'Frame trend — {mframe}.')
    d.fig(14, f'Frame trend — {qn}.')
    Wf, Wfq = lambda l: c.W('frame_m', l), lambda l: c.W('frame_q', l)
    pbw, pbq, ecw, ecq, msw, huw, huq = (Wf('Political blame'), Wfq('Political blame'), Wf('Economic/sanctions'),
                                         Wfq('Economic/sanctions'), Wf('Military/strategic'), Wf('Humanitarian'),
                                         Wfq('Humanitarian'))
    A(ecw.index(max(ecw)) == 1 and ecq.index(max(ecq)) == 1 and pbw.index(max(pbw)) == 5)
    pbcnt = [R['weekly']['frame_m_count']['Political blame'][w] for w in c.wk]
    diffs = [a - b for a, b in zip(pbw, pbq)]
    if not FZ:
        qpk = c.peak('frame_q', 'Political blame')
        d.para(f'Political blame rises from {pc(pbw[0])} of videos in the first week to {min(pbw[2:4]):.1f}–'
               f'{max(pbw[2:4]):.1f}% in mid-to-late March and peaks at {pc(pbw[5])} in April 6–12; Qwen peaks in '
               f'{qpk[0]} ({pc(qpk[1])}), with April 6–12 close behind ({pc(pbq[5])}). Both models show a similar pattern '
               f'(weekly correlation {R["week_corr"]["Political blame"]:.2f}), with Mistral {min(diffs):.0f}–'
               f'{max(diffs):.0f} percentage points higher than Qwen every week. Economic/sanctions peaks in March 9–15 '
               f'in both models (Mistral {pc(ecw[1])}, Qwen {pc(ecq[1])}). Military/strategic is low in the first two '
               f'weeks ({pc(msw[0])} and {pc(msw[1])}) and in the final week ({pc(msw[-1])}), and around 20–21% in most '
               f'weeks from late March to mid-April. Humanitarian framing declines across the period in both models '
               f'(Mistral {pc(huw[0])} to {pc(huw[-1])}, Qwen {pc(huq[0])} to {pc(huq[-1])}).')
    else:
        qpk = c.peak('frame_q', 'Political blame')
        A(qpk[0] == WK[5])
        d.para(f'Political blame rises from {pc(pbw[0])} in the first two weeks to {pc(pbw[2])} by mid-March and peaks at '
               f'{pc(pbw[5])} in April 6–12; Qwen shows the same peak week ({pc(qpk[1])}). Economic/sanctions peaks in '
               f'March 9–15 in both models (Mistral {pc(ecw[1])}, Qwen {pc(ecq[1])}). Military/strategic stays between '
               f'{min(msw):.1f}% and {max(msw):.1f}% per week in Mistral, peaking in {c.peak("frame_m", "Military/strategic")[0]}, '
               f'while Qwen assigns it {min(Wfq("Military/strategic")):.1f}–{max(Wfq("Military/strategic")):.1f}% per '
               f'week. Humanitarian framing declines across the period in both models (Mistral {pc(huw[0])} to '
               f'{pc(huw[-1])}, Qwen {pc(huq[0])} to {pc(huq[-1])}).')
    d.fig(15, f'Remaining frame labels — {mframe}.')
    d.fig(16, f'Remaining frame labels — {qn}.')
    hc, rel, mis, sat, mc = Wf('Historical context'), Wf('Religious framing'), Wf('Misinformation'), Wf('Satire/meme'), Wf('Media criticism')
    misq = Wfq('Misinformation')
    if not FZ:
        A(rel.index(max(rel)) == 0 and mis.index(max(mis)) == 5)
        d.para(f'Historical context is highest in the first two weeks ({pc(hc[0])} and {pc(hc[1])}) and then stays at '
               f'about 2–3%; its decline is tested in Section 3.5. Religious framing moves between {min(rel):.1f}% and '
               f'{max(rel):.1f}%, Misinformation between {min(mis):.1f}% and {max(mis):.1f}% (a label with no gold rows) and '
               f'Satire/meme between {min(sat):.1f}% and {max(sat):.1f}%. Weekly movements of these small labels are 1–3 percentage points on a few hundred videos, so they are not read as findings; Section 3.5 tests the trends.')
    else:
        A(rel.index(max(rel)) == 0 and mc.index(max(mc)) == 5)
        d.para(f'Religious framing moves between {min(rel):.1f}% and {max(rel):.1f}% per week, Media criticism between '
               f'{min(mc):.1f}% and {max(mc):.1f}% and Satire/meme between {min(sat):.1f}% and {max(sat):.1f}%. '
               f'Weekly movements of these small labels are 1–3 percentage points on a few hundred videos, so they are not read as findings; Section 3.5 tests the trends. Qwen zero-shot assigns Misinformation to {min(misq):.1f}–'
               f'{max(misq):.1f}% of videos every week, against {min(mis):.1f}–{max(mis):.1f}% for Mistral; with no '
               f'Misinformation rows in the gold standard, the Qwen figure cannot be checked.')

    # 3.4 cross-model
    d.h2('3.4  Cross-Model Comparison')
    d.fig(17, f'Fear weekly trend — {msent} vs {qn}.')
    d.fig(18, f'Anger weekly trend — {msent} vs {qn}.')
    fm, fq, am, aq = Ws('Fear'), Wq('Fear'), Ws('Anger'), Wq('Anger')
    wc = R['week_corr']
    if S == 'best':
        ratio = [b / a for a, b in zip(fm, fq)]
        A(fq.index(max(fq)) == 5 and aq.index(max(aq)) == 5)
        d.para(f'Both models put the peak share of Fear and Anger in April 6–12. Qwen assigns Fear at '
               f'{min(ratio):.1f}–{max(ratio):.1f} times Mistral\'s rate each week — Mistral ranges from {min(fm):.1f}% to '
               f'{max(fm):.1f}% per week and Qwen from {min(fq):.1f}% to {max(fq):.1f}% — but the two weekly series move '
               f'together (correlation {wc["Fear"]:.2f}). Anger levels are close between the models (correlation '
               f'{wc["Anger"]:.2f}) and almost identical at the peak (Mistral {pc(am[5])}, Qwen {pc(aq[5])} in April 6–12). '
               f'Anger has only {R["gold_counts"]["t1"]["Anger"]} gold rows, so agreement between models does not show '
               f'that Anger is classified accurately.')
    elif S == 'zs':
        A(aq.index(max(aq)) == 5)
        d.para(f'Both models put the Anger peak in April 6–12 (Mistral {pc(am[5])}, Qwen {pc(aq[5])}). For Fear, Qwen '
               f'assigns {min(fq):.1f}–{max(fq):.1f}% of videos per week against {min(fm):.1f}–{max(fm):.1f}% for Mistral, '
               f'and its Fear peak falls in {c.peak("sent_q", "Fear")[0]}. The weekly series are still correlated (Fear '
               f'{wc["Fear"]:.2f}, Anger {wc["Anger"]:.2f}), so the two models agree more on direction than on level.')
    else:
        A(fq.index(max(fq)) == 5 and aq.index(max(aq)) == 5)
        d.para(f'In the few-shot setting the two models agree closely on Fear: Mistral ranges from {min(fm):.1f}% to '
               f'{max(fm):.1f}% per week and Qwen from {min(fq):.1f}% to {max(fq):.1f}%, both peak in April 6–12, and the '
               f'weekly series are strongly correlated ({wc["Fear"]:.2f}). Anger also peaks in April 6–12 in both models '
               f'(correlation {wc["Anger"]:.2f}), although Mistral assigns it at a higher level ({min(am):.1f}–'
               f'{max(am):.1f}% against {min(aq):.1f}–{max(aq):.1f}%).')

    # 3.5 trend tests
    d.h2('3.5  Trend Tests')
    d.para('The Cochran–Armitage test checks whether a label\'s weekly share rises or falls steadily across the nine '
           'weeks (negative z = decline). It was run on all videos and again without the month-query videos, which '
           'cluster at the end of each month.')
    rows = []
    tests = [('Solidarity', 'Solidarity_m', 'sent_m', 'Solidarity'), ('Solidarity (Qwen)', 'Solidarity_q', 'sent_q', 'Solidarity'),
             ('Humanitarian', 'Humanitarian_m', 'frame_m', 'Humanitarian'),
             ('Humanitarian (Qwen)', 'Humanitarian_q', 'frame_q', 'Humanitarian'),
             ('Political blame', 'Political_m', 'frame_m', 'Political blame'), ('Fear', 'Fear_m', 'sent_m', 'Fear'),
             ('Anger', 'Anger_m', 'sent_m', 'Anger'), ('Humour', 'Humour_m', 'sent_m', 'Humour'),
             ('Satire/meme', 'Satire_m', 'frame_m', 'Satire/meme'),
             ('Historical context', 'Historical_m', 'frame_m', 'Historical context')]
    for lab, key, tag, l in tests:
        w = c.W(tag, l)
        t, t2 = c.tt(key), c.tt(key + '_nomonth')
        rows.append([lab, pc(w[0]), pc(w[-1]), f'{t["z"]:.2f}', pv(t['p']),
                     f'{t2["z"]:.2f}', pv(t2['p'])])
    d.table(['Label', 'First week', 'Final week', 'z (all)', 'p (all)', 'z (no month)', 'p (no month)'], rows,
            [2300, 1100, 1100, 1000, 1200, 1250, 1300], size=18)
    d.caption(f'Table 4. Cochran–Armitage trend tests on weekly shares — {M} unless marked Qwen.')
    ang_t, ang_t2 = c.tt('Anger_m'), c.tt('Anger_m_nomonth')
    hum_t1, hum_t2 = c.tt('Humour_m'), c.tt('Humour_m_nomonth')
    pol_t = c.tt('Political_m')
    A(sol_t['p'] < 0.001 and c.tt('Solidarity_q')['p'] < 0.001 and c.tt('Solidarity_m_nomonth')['p'] < 0.001)
    A(hum_t['p'] < 0.001 and c.tt('Humanitarian_m_nomonth')['p'] < 0.001 and pol_t['p'] < 0.001)
    A(ang_t['p'] > 0.05 and ang_t2['p'] < 0.05)
    fear_t = c.tt('Fear_m')
    A(fear_t['p'] > 0.05)
    A(hum_t1['p'] > 0.05 and hum_t2['p'] < 0.05 and c.tt('Political_m_nomonth')['p'] < 0.001)
    hct, hct2 = c.tt('Historical_m'), c.tt('Historical_m_nomonth')
    an4, an5 = R['weekly']['sent_m_nomonth']['Anger'][c.wk[4]], R['weekly']['sent_m_nomonth']['Anger'][c.wk[5]]
    d.para(f'The Solidarity and Humanitarian declines and the Political blame rise are the clearest trends in the '
           f'classifier output. All three hold without the month-query videos, and the two declines also hold in Qwen. '
           f'With classification error included (Section 8.2), the Political blame rise and the Humanitarian decline '
           f'remain, while the Solidarity decline is no longer confirmed. '
           + (f'Historical context also declines ({fp(hct["p"])}; {fp(hct2["p"])} without month-query videos). '
              if hct['p'] < 0.05 else '')
           + f'Fear shows no steady trend ({fp(fear_t["p"])}); it moves up and down with the events instead. Anger and '
             f'Humour show a rise only when the month-query videos are removed (Anger {fp(ang_t["p"])} with them, '
             f'{fp(ang_t2["p"])} without; Humour {fp(hum_t1["p"])} and {fp(hum_t2["p"])}), so these trends are weak. '
           + (f'Without the month-query videos, Anger in the March 30 window ({pc(an4)}) is as high as in the April 6 '
              f'window ({pc(an5)}): the late-March and early-April windows both carry high Anger.'
              if an4 >= an5 - 1 else
              f'Without the month-query videos, Anger in the March 30 window rises to {pc(an4)}, but the April 6 window '
              f'remains the peak ({pc(an5)}).'))

    # ── 4. Co-occurrence ─────────────────────────────────────────────────────
    d.h1('4. Sentiment × Frame Co-Occurrence')
    src = ('Mistral zero-shot sentiment × Mistral few-shot frame' if S == 'best' else f'both tiers from the same Mistral '
           f'{ss} run')
    d.para(f'The co-occurrence analysis shows which narrative frames go with each sentiment label ({src}). Two measures '
           f'are used. The row share is the percentage of a sentiment\'s videos that carry a frame; it favours the large '
           f'frames. Lift divides that share by the frame\'s share across all videos: a lift of 3 means the frame is three '
           f'times as common with this sentiment as overall. A pairing can have a high share and modest lift (Anger → '
           f'Political blame) or a small share and very high lift (Disgust → Religious framing). Fisher\'s exact test '
           f'checks whether a pairing is more common than chance.')
    d.fig(19, f'Co-occurrence — raw counts. {msent} sentiment × {mframe} frame.')
    d.fig(20, 'Co-occurrence — row-normalised (% of each sentiment assigned to each frame).')
    d.fig(21, 'Co-occurrence — lift (row share ÷ overall frame share). Cells with fewer than 5 videos left blank; red = '
              'more common than overall, blue = less common.')
    d.h2('4.1  Dominant Co-Occurrence Patterns')

    def cob(s, f, extra=''):
        v = c.co(s, f)
        cv = R['cooc_ci'].get(s, {}).get(f)
        q = c.co(s, f, 'q')['pct']
        civ = f' [95% CI: {cv[1]:.1f}–{cv[2]:.1f}%]' if cv else ''
        return (f'{pc(v["pct"])}{civ} in Mistral ({q:.0f}% in Qwen); lift {v["lift"]:.1f}, Fisher {fp(v["p"])}.'
                + (' ' + extra if extra else ''))

    A(c.cotop('Anger') == 'Political blame' and c.cotop('Anger', 'q') == 'Political blame')
    d.bullet(cob('Anger', 'Political blame',
                 'Political blame is the top frame for Anger in both models.'), 'Anger → Political blame: ')
    hs = c.co('Humour', 'Satire/meme')
    A(c.cotop('Humour') == 'Satire/meme')
    qh = c.cotop('Humour', 'q', 2, skip_unclear=False)
    lifts = {(s_, f_): c.co(s_, f_)['lift'] for s_ in SENT9 for f_ in FR10
             if c.co(s_, f_)['lift'] and c.co(s_, f_)['pct'] * R['counts']['sent'].get(s_, 0) / 100 >= 5}
    best_pair = max(lifts, key=lifts.get)
    q1, q2 = (c.co('Humour', f_, 'q')['pct'] for f_ in qh)
    d.bullet(cob('Humour', 'Satire/meme',
                 (f'The strongest association in the data by lift, and it involves the most reliable emotional label. '
                  if best_pair == ('Humour', 'Satire/meme') else
                  f'One of the strongest associations by lift, and it involves the most reliable emotional label. ')
                 + (f'Qwen splits Humour between {qh[0]} ({q1:.0f}%) and {qh[1]} ({q2:.0f}%).' if abs(q1 - q2) < 5
                    else f'Qwen does not reproduce it: its top frame for Humour is {qh[0]} ({q1:.0f}%).'
                    if qh[0] != 'Satire/meme' else f'Qwen agrees ({q1:.0f}%).')),
             'Humour → Satire/meme: ')
    A(c.cotop('Sadness') == 'Humanitarian' and c.cotop('Solidarity') == 'Humanitarian')
    sa, so_ = c.co('Sadness', 'Humanitarian'), c.co('Solidarity', 'Humanitarian')
    d.bullet(f'Sadness {pc(sa["pct"])} (lift {sa["lift"]:.0f}) and Solidarity {pc(so_["pct"])} (lift {so_["lift"]:.0f}) in '
             f'Mistral; Qwen {c.co("Sadness", "Humanitarian", "q")["pct"]:.0f}% and '
             f'{c.co("Solidarity", "Humanitarian", "q")["pct"]:.0f}%. Both Fisher {fp(max(sa["p"], so_["p"]))}.', 'Sadness and Solidarity → Humanitarian: ')
    fear_top = c.cotop('Fear')
    fqt = c.cotop('Fear', 'q')
    fm_ = c.co('Fear', 'Military/strategic')
    if fear_top == 'Military/strategic':
        d.bullet(cob('Fear', 'Military/strategic', 'Fear goes with Military/strategic framing in both models, but '
                                                    'Fear has no gold rows, so this is a directional finding only.'),
                 'Fear → Military/strategic: ')
    else:
        d.bullet(f'Fear is coded Political blame in {pc(c.co("Fear", "Political blame")["pct"])} of cases and '
                 f'Military/strategic in {pc(fm_["pct"])}; Qwen reverses the order ({fqt} '
                 f'{c.co("Fear", fqt, "q")["pct"]:.1f}%). By lift, Military/strategic is the stronger association '
                 f'({fm_["lift"]:.1f} against {c.co("Fear", "Political blame")["lift"]:.1f}), consistent with the other '
                 f'settings; the high Political blame share comes from the Political blame over-assignment.',
                 'Fear → Political blame or Military/strategic: ')
        A(fqt == 'Military/strategic')
    dr = c.co('Disgust', 'Religious framing')
    d.bullet(f'Religious framing {pc(dr["pct"])} of Disgust videos (lift {dr["lift"]:.0f}, Fisher {fp(dr["p"])}), next to '
             f'Political blame ({pc(c.co("Disgust", "Political blame")["pct"])}). Disgust rests on {R["counts"]["sent"]["Disgust"]} videos.',
             'Disgust → Religious framing: ')
    frt = c.cotop('Frustration')
    fqtop = c.cotop('Frustration', 'q')
    other_set = {s: Ctx(s).cotop('Frustration') for s in ('best', 'zs', 'fs')}
    stable = len(set(other_set.values())) == 1
    d.bullet(f'{frt} is the top frame ({pc(c.co("Frustration", frt)["pct"])}, lift '
             f'{c.co("Frustration", frt)["lift"]:.1f}); '
             + (f'Qwen agrees ({c.co("Frustration", fqtop, "q")["pct"]:.0f}%)' if fqtop == frt else
                f'Qwen pairs Frustration mainly with {fqtop} ({c.co("Frustration", fqtop, "q")["pct"]:.0f}%)')
             + ('' if stable else (', but the top frame changes with the prompt setting' if fqtop == frt else
                                   ', and the top frame changes with the prompt setting'))
             + '. This finding depends on the model and setting and should not be reported as a definitive result.',
             f'Frustration → {frt}: ')
    if not SZ:
        cu = c.cotop('Curiosity')
        d.bullet(f'{pc(c.co("Curiosity", cu)["pct"])} {cu}. Curiosity is the largest emotional label in this setting but '
                 f'is mostly a false positive on the gold sample, so this pairing is not interpreted.',
                 f'Curiosity → {cu}: ')
    pr = {l: c.prec(l) for l in ('Anger', 'Fear', 'Solidarity', 'Sadness', 'Humour')}
    d.para(f'Both labels in each pairing come from the classifier, so these patterns show how Mistral links its own labels. '
           f'They describe content only as far as the labels are correct. On the gold sample, '
           f'{pr["Humour"][0]} of {pr["Humour"][1]} Humour predictions were confirmed, but only {pr["Anger"][0]} of '
           f'{pr["Anger"][1]} Anger, {pr["Solidarity"][0]} of {pr["Solidarity"][1]} Solidarity and {pr["Fear"][0]} of '
           f'{pr["Fear"][1]} Fear predictions. Humour → Satire/meme is therefore the best-supported pairing; pairings that '
           f'involve Anger, Fear, Sadness or Solidarity are reported as classifier patterns, not as how creators combine '
           f'emotion and framing.')

    # ── 5. Language ──────────────────────────────────────────────────────────
    d.h1('5. Language Community Comparison')
    ln = R['lang_n']
    lc = R['lang_chi']
    d.para(f'The dataset covers 6 major language communities: English ({ln["English"]:,} videos, '
           f'{ln["English"] / 58.48:.1f}%), Indonesian ({ln["Indonesian"]}), German ({ln["German"]}), Spanish '
           f'({ln["Spanish"]}), French ({ln["French"]}), and Catalan ({ln["Catalan"]}). Section 1.4 applies here: high '
           f'Unclear rates in German and Catalan content mostly reflect short or hashtag-only posts that annotators also '
           f'could not frame. Across these six languages, the sentiment mix differs (χ² {fp(lc["sentiment_label"]["p"])}, '
           f'Cramér\'s V {lc["sentiment_label"]["V"]:.2f}) and the frame mix differs more strongly (χ² '
           f'{fp(lc["frame_label"]["p"])}, V {lc["frame_label"]["V"]:.2f}); much of the frame difference is the Unclear '
           f'share.')
    d.h2('5.1  Sentiment by Language')
    d.fig(22, f'Sentiment by language community — {msent}.')
    d.fig(23, f'Sentiment by language — {msent} (solid bars) vs {qn} (hatched bars) cross-model check.', width=6.2)
    d.h2('5.2  Frame by Language')
    d.fig(24, f'Frame by language community — {mframe}.')
    d.fig(25, 'Religious framing by language community with 95% bootstrap confidence intervals.')
    d.h2('5.3  Community-Level Findings')
    L = lambda g, lab, k='frame', m='m': c.sh('lang', k, g, lab, m)
    lci = R['lang_ci']
    ihc = c.pt('Indonesian|English|Historical context')
    A(c.top('lang', 'frame', 'Religious framing') == 'Indonesian' and id_rel['p'] < 0.001)
    A(L('Indonesian', 'Religious framing', m='q') > L('English', 'Religious framing', m='q'))
    hc_rank = c.rank('lang', 'frame', 'Historical context', 'Indonesian')
    d.para(f'Religious framing is {ci(lci["Religious framing"]["Indonesian"])}, the highest of any language community '
           f'and {times(id_rel["pa"], id_rel["pb"])} English ({ci(lci["Religious framing"]["English"])}; Fisher {fp(id_rel["p"])}). Qwen shows '
           f'the same direction ({pc(L("Indonesian", "Religious framing", m="q"))} vs '
           f'{pc(L("English", "Religious framing", m="q"))}). Historical context is also high ({pc(ihc["pa"])} vs '
           f'{pc(ihc["pb"])} for English, {fp(ihc["p"])}'
           + (', second to French' if hc_rank == 2 else ', the highest of the six' if hc_rank == 1 else '')
           + f'). With classification error included (Section 8.2), the Religious framing difference holds (corrected '
             f'{pc(c.ec("Religious framing: Indonesian vs English")["corr_a"])} vs '
             f'{pc(c.ec("Religious framing: Indonesian vs English")["corr_b"])}, '
             f'{fp(c.ec("Religious framing: Indonesian vs English")["p"])}), but the Historical context difference is not '
             f'confirmed ({fp(c.ec("Historical context: Indonesian vs English")["p"])}). All 308 posts were classified '
             f'from English translations, and the gold sample has only one Indonesian Religious framing row, so this is '
             f'a difference in the classifier output, not a validated statement about Indonesian creators.',
           'Indonesian community (308 videos): ')
    ffe, fan, fhc = c.pt('French|English|Fear'), c.pt('French|English|Anger'), c.pt('French|English|Historical context')
    A(ffe['p'] > 0.05 and fan['p'] > 0.05)
    fr_mil = L('French', 'Military/strategic')
    fr_txt = (f'Fear {pc(ffe["pa"])} and Anger {pc(fan["pa"])}, against {pc(ffe["pb"])} and {pc(fan["pb"])} for '
              f'English; neither difference is significant (Fisher {fp(ffe["p"])} and {fp(fan["p"])}). ')
    if not FZ:
        A(c.rank('lang', 'frame', 'Military/strategic', 'French') == 1 and c.rank('lang', 'frame', 'Historical context', 'French') == 1)
        fr_txt += (f'Military/strategic ({pc(fr_mil)}) and Political blame ({pc(L("French", "Political blame"))}) are '
                   f'level. Historical context is {pc(fhc["pa"])} against {pc(fhc["pb"])} for English ({fp(fhc["p"])}), but '
                   f'this is not confirmed once classification error is included '
                   f'({fp(c.ec("Historical context: French vs English")["p"])}; Historical context has 3 gold rows).')
    else:
        A(c.rank('lang', 'frame', 'Political blame', 'French') == 1)
        fr_txt += (f'Political blame is {pc(L("French", "Political blame"))}, against {pc(L("English", "Political blame"))} '
                   f'for English. Historical context is {pc(fhc["pa"])} against {pc(fhc["pb"])} for English ({fp(fhc["p"])}).')
    d.para(fr_txt, 'French community (101 videos): ')
    en_top = sorted(FR10, key=lambda l: -L('English', l))[:2]
    d.para(f'Provides the baseline distribution — Fear {pc(L("English", "Fear", "sent"))}, Anger '
           f'{pc(L("English", "Anger", "sent"))}, with {en_top[0]} ({pc(L("English", en_top[0]))}) and {en_top[1]} '
           f'({pc(L("English", en_top[1]))}) as the top frames. Historical context ({pc(L("English", "Historical context"))}) '
           f'and Religious framing ({pc(L("English", "Religious framing"))}) are both low.',
           f'English community ({ln["English"]:,} videos): ')
    sm = c.pt('Spanish|English|Misinformation')
    if not FZ:
        A(c.rank('lang', 'frame', 'Misinformation', 'Spanish') == 1)
        d.para(f'Misinformation is {pc(sm["pa"])} against {pc(sm["pb"])} for English ({fp(sm["p"])}), from {sm["a"]} '
               f'videos. Misinformation has no gold rows and none of its {c.prec("Misinformation")[1]} predictions on the '
               f'gold sample was confirmed, so this result cannot be validated.', 'Spanish community (115 videos): ')
    else:
        A(c.rank('lang', 'frame', 'Satire/meme', 'Spanish') == 1)
        d.para(f'No label differs clearly from English; most shares rest on fewer than 10 videos.',
               'Spanish community (115 videos): ')
    d.para(f'{pc(L("German", "Unclear"))} and {pc(L("Catalan", "Unclear"))} Unclear frame, and '
           f'{pc(L("German", "Neutral", "sent"))} and {pc(L("Catalan", "Neutral", "sent"))} Neutral sentiment. These rates '
           f'mostly reflect posts with little classifiable text, so no substantive interpretation is made.',
           'German (211 videos) and Catalan (63 videos) communities: ')

    # ── 6. Region ────────────────────────────────────────────────────────────
    d.h1('6. Regional Analysis')
    rn = R['reg_n']
    rc = R['reg_chi']
    d.para('The dataset covers 8 major regions: ' + ', '.join(f'{g} ({rn[g]:,}, {rn[g] / 58.48:.1f}%)' for g in REG8)
           + f'. The Section 1.4 findings apply — high Unclear rates in Pakistan and Germany mostly reflect non-English '
             f'posts with little classifiable text. Across the eight regions the sentiment mix differs (χ² '
             f'{fp(rc["sentiment_label"]["p"])}, V {rc["sentiment_label"]["V"]:.2f}) and so does the frame mix (χ² '
             f'{fp(rc["frame_label"]["p"])}, V {rc["frame_label"]["V"]:.2f}); both effects are small.')
    d.h2('6.1  Sentiment by Region')
    d.fig(26, f'Sentiment by region — {msent}.')
    d.fig(27, f'Sentiment by region — {msent} (solid bars) vs {qn} (hatched bars) cross-model check.', width=6.2)
    d.h2('6.2  Frame by Region')
    d.fig(28, f'Frame by region — {mframe}.')
    d.fig(29, 'Religious framing by region with 95% bootstrap confidence intervals.')
    d.fig(30, 'Historical context framing by language (left) and region (right) with 95% bootstrap confidence intervals.')
    d.h2('6.3  Regional Findings')
    G = lambda g, lab, k='frame', m='m': c.sh('reg', k, g, lab, m)
    rci = R['reg_ci']
    fa = rci['Fear+Anger']
    r50 = R['reg50']

    def top50(lab, key):
        return sorted(r50, key=lambda r: -r50[r][key].get(lab, 0))

    usde, ngus = c.pt('US|DE|Fear'), c.pt('NG|US|Fear')
    idus_r, ngus_r, idus_h = c.pt('ID|US|Religious framing'), c.pt('NG|US|Religious framing'), c.pt('ID|US|Historical context')
    aeus = c.pt('AE|US|Anger')
    paras = []
    # United States
    us_fr = sorted(FR10, key=lambda l: -G('US', l))[:3]
    if SZ:
        c.hi('reg', 'sent', 'US', 'Fear')
        A(ngus['p'] > 0.05 and usde['p'] > 0.05)
        txt = (f'Fear {pc(G("US", "Fear", "sent"))} and Anger {pc(G("US", "Anger", "sent"))}; combined Fear + Anger is '
               f'{ci(fa["US"])}. The US has the highest Fear share of the eight regions, but not significantly above '
               f'Germany ({fp(usde["p"])}) or Nigeria ({fp(ngus["p"])}), and Fear has no gold rows. ')
    else:
        txt = (f'Fear {pc(G("US", "Fear", "sent"))} and Anger {pc(G("US", "Anger", "sent"))}. Frustration '
               f'({pc(G("US", "Frustration", "sent"))}) and Curiosity ({pc(G("US", "Curiosity", "sent"))}) are high, but '
               f'these are the two labels most over-assigned in this setting ({c.prec("Frustration")[0]} of '
               f'{c.prec("Frustration")[1]} and {c.prec("Curiosity")[0]} of {c.prec("Curiosity")[1]} predictions confirmed '
               f'on the gold sample). ')
    txt += 'The main frames are ' + lst(f'{l} ({pc(G("US", l))})' for l in us_fr) + '.'
    paras.append(('US', txt))
    # United Kingdom
    c.hi('reg', 'sent', 'GB', 'Solidarity')
    A(top50('Solidarity', 's')[0] == 'GB')
    gbq_rank = c.rank('reg', 'sent', 'Solidarity', 'GB', 'q')
    gbc = c.ec('Solidarity: GB vs US')
    txt = (f'The highest Solidarity rate of the eight regions in the classifier output ({ci(rci["Solidarity"]["GB"])}), '
           f'{times(gb_sol["pa"], gb_sol["pb"])} the US rate ({pc(gb_sol["pb"])}; Fisher {fp(gb_sol["p"])}). '
           + (f'Qwen also gives GB its highest Solidarity rate ({pc(G("GB", "Solidarity", "sent", "q"))}). ' if gbq_rank == 1
              else f'Qwen also rates GB high ({pc(G("GB", "Solidarity", "sent", "q"))}), though it puts '
                   f'{c.top("reg", "sent", "Solidarity", "q")} slightly higher. ')
           + f'The difference is not confirmed once classification error is included (corrected {pc(gbc["corr_a"])} vs '
             f'{pc(gbc["corr_b"])}, {fp(gbc["p"])}): only {c.prec("Solidarity")[0]} of {c.prec("Solidarity")[1]} '
             f'Solidarity predictions {"was" if c.prec("Solidarity")[0] == 1 else "were"} confirmed on the gold sample. Humanitarian framing is {pc(G("GB", "Humanitarian"))}.')
    A(gbc['p'] > 0.05)
    paras.append(('GB', txt))
    # Pakistan
    c.hi('reg', 'frame', 'PK', 'Unclear')
    A(min(fa, key=lambda g: fa[g][0]) == 'PK')
    txt = (f'The highest Unclear frame rate ({pc(G("PK", "Unclear"))} Mistral, {pc(G("PK", "Unclear", m="q"))} Qwen) and '
           f'the lowest Fear + Anger ({ci(fa["PK"])}). English-language Pakistani posts have {pc(ne["PK"]["unc_en"])} '
           f'Unclear against {pc(ne["PK"]["unc_ne"])} for non-English posts. As Section 1.4 shows, this mostly reflects '
           f'short or hashtag-only non-English posts, so Pakistani results are not interpreted further.')
    paras.append(('PK', txt))
    # Nigeria
    A(c.rank('reg', 'frame', 'Religious framing', 'NG') == 2 and ngus_r['p'] < 0.01)
    ngc = c.ec('Religious framing: NG vs US')
    A(ngc['p'] < 0.05)
    nf = (f'Religious framing is {ci(rci["Religious framing"]["NG"])}, second of the eight regions after Indonesia and above '
          f'the US ({pc(ngus_r["pb"])}; {fp(ngus_r["p"])}). The difference holds with classification error included '
          f'(corrected {pc(ngc["corr_a"])} vs {pc(ngc["corr_b"])}, {fp(ngc["p"])}). {ne["NG"]["n_en"]} of the '
          f'{rn["NG"]} Nigerian posts are in English, so this is not a translation effect; it is a difference in how often '
          f'posts use religious framing, not evidence about how Nigerians interpret the crisis. ')
    top2 = sorted(FR10, key=lambda l: -G('NG', l))[:2]
    if SZ:
        txt = nf + f'{top2[0]} ({pc(G("NG", top2[0]))}) and {top2[1]} ({pc(G("NG", top2[1]))}) are the main frames.'
    else:
        c.hi('reg', 'sent', 'NG', 'Fear')
        A(ngus['p'] < 0.05 and c.top('reg', 'sent', 'Fear', 'q') == 'NG')
        txt = (f'Fear {ci(rci["Fear"]["NG"])}, the highest of the eight regions (against {pc(ngus["pb"])} for the US, Fisher '
               f'{fp(ngus["p"])}), and Qwen also gives Nigeria its highest Fear rate ({pc(G("NG", "Fear", "sent", "q"))}). '
               f'Fear has no gold rows, so this cannot be confirmed. ' + nf)
    paras.append(('NG', txt))
    # Indonesia
    c.hi('reg', 'frame', 'ID', 'Religious framing')
    c.hi('reg', 'frame', 'ID', 'Historical context')
    idc = c.ec('Religious framing: ID vs US')
    A(idc['p'] < 0.05)
    my = r50.get('MY', {'f': {}})['f'].get('Religious framing', 0)
    txt = (f'Religious framing ({ci(rci["Religious framing"]["ID"])}; {fp(idus_r["p"])} against the US) and Historical '
           f'context ({pc(idus_h["pa"])} vs {pc(idus_h["pb"])} for the US, {fp(idus_h["p"])}) are the highest of the eight '
           f'largest regions')
    if my > G('ID', 'Religious framing'):
        txt += f'; among smaller regions only Malaysia ({r50["MY"]["n"]} videos) has a higher Religious framing rate ({pc(my)})'
    txt += (f'. The Religious framing difference holds with classification error included (corrected {pc(idc["corr_a"])} '
            f'vs {pc(idc["corr_b"])}, {fp(idc["p"])}); the Historical context difference between Indonesian and English posts '
            f'does not (Section 8.2). '
            f'{ne["ID"]["n_ne"]} of the {rn["ID"]} videos are non-English, so the caveats in Section 5.3 apply.')
    paras.append(('ID', txt))
    # UAE
    ae_top = sorted(FR10, key=lambda l: -G('AE', l))[:2]
    fa_rank = sorted(fa, key=lambda g: -fa[g][0])
    A(fa_rank.index('AE') <= 2)
    txt = (f'Combined Fear and Anger is {ci(fa["AE"])}. Anger ({pc(G("AE", "Anger", "sent"))}) does not differ from the US '
           f'({pc(aeus["pb"])}; {fp(aeus["p"])}). {ae_top[0]} ({pc(G("AE", ae_top[0]))}) and {ae_top[1]} '
           f'({pc(G("AE", ae_top[1]))}) are the main frames. Religious framing is {ci(rci["Religious framing"]["AE"])}.')
    paras.append(('AE', txt))
    # France
    fr_top = sorted(FR10, key=lambda l: -G('FR', l))[:2]
    txt = (f'Fear {pc(G("FR", "Fear", "sent"))}, Anger {pc(G("FR", "Anger", "sent"))}. {fr_top[0]} '
           f'({pc(G("FR", fr_top[0]))}) and {fr_top[1]} ({pc(G("FR", fr_top[1]))}) are the two main frames, and Historical '
           f'context is {pc(G("FR", "Historical context"))}. {ne["FR"]["n_ne"]} of the {rn["FR"]} videos are non-English.')
    paras.append(('FR', txt))
    # Germany
    A(usde['p'] > 0.5)
    if SZ:
        A(c.rank('reg', 'sent', 'Fear', 'DE') == 2)
        txt = (f'Fear {ci(rci["Fear"]["DE"])}, second of the eight regions after the US; the difference is within sampling '
               f'variation (Fisher {fp(usde["p"])}). ')
    else:
        txt = (f'Fear {pc(G("DE", "Fear", "sent"))}, {c.rank("reg", "sent", "Fear", "DE")}th of the eight — Germany\'s '
               f'high Fear rate in the zero-shot setting does not appear here. ')
    txt += (f'The Unclear frame rate is {pc(G("DE", "Unclear"))}, falling to {pc(ne["DE"]["unc_en"])} for English-language '
            f'German posts. Treat findings tentatively.')
    paras.append(('DE', txt))
    for g, t in paras:
        d.para(t, f'{REGNAME[g]} ({rn[g]:,} videos): ')

    # ── 7. Engagement ────────────────────────────────────────────────────────
    d.h1('7. Engagement Analysis')
    kw = R['kw']
    d.para(f'Median likes and views are used throughout to avoid distortion from viral outliers, with 95% bootstrap '
           f'intervals shown as error bars; shares and comments are reported as means because their medians are 0–2 for '
           f'every label. Each label is compared with all other labelled videos using the Mann–Whitney U test. '
           f'Engagement differs across sentiment labels (Kruskal–Wallis, likes {fp(kw["sent_like_count"]["p"])}, views '
           f'{fp(kw["sent_view_count"]["p"])}) and across frames (likes {fp(kw["frame_like_count"]["p"])}, views '
           f'{fp(kw["frame_view_count"]["p"])}). Groups are defined by the classifier\'s labels, so "Solidarity videos" means '
           f'videos Mistral labels Solidarity. For most emotional labels few predictions were confirmed on the gold sample '
           f'(Section 1.2), so sentiment-based engagement results describe the classifier\'s labels. Frame-based results '
           f'rest on better-classified labels.')
    d.h2('7.1  Engagement by Sentiment')
    d.fig(31, f'Median likes by sentiment with 95% bootstrap CIs — {msent}.')
    d.fig(32, f'Median views by sentiment with 95% bootstrap CIs — {msent}.')
    SE = SENT9 + ['Neutral']
    big = [l for l in SE if c.es(l)['n'] >= 100]
    ml = sorted(SE, key=lambda l: -c.es(l)['med_likes'])
    A(ml[0] == 'Solidarity')
    qml = max(SE, key=lambda l: R['eng_sent_q'][l]['med_likes'])
    s_ = c.es('Solidarity')
    d.bullet(f'Videos labelled Solidarity receive the highest median likes ({num(s_["med_likes"])} [{nr(s_["likes_ci"][1])}–'
             f'{nr(s_["likes_ci"][2])}]), about twice the next label ({ml[1]}, {num(c.es(ml[1])["med_likes"])}); the '
             f'difference from other videos is significant ({fp(s_["mw_likes_p"])}). Its median views '
             f'({num(s_["med_views"])}) are not ({fp(s_["mw_views_p"])}). '
             + (f'Qwen {qs} also gives Solidarity the highest median likes '
                f'({num(R["eng_sent_q"]["Solidarity"]["med_likes"])}).' if qml == 'Solidarity' else
                f'Qwen puts {qml} first ({num(R["eng_sent_q"][qml]["med_likes"])}).'))
    mv = sorted(SE, key=lambda l: -c.es(l)['med_views'])
    mvb = sorted(big, key=lambda l: -c.es(l)['med_views'])
    fe = c.es('Fear')
    A(mvb[0] == 'Fear' and fe['mw_views_p'] < 0.001)
    qmv = max([l for l in SE if R['eng_sent_q'][l]['n'] >= 100], key=lambda l: R['eng_sent_q'][l]['med_views'])
    d.bullet((f'Videos labelled Fear have the highest median views ({num(fe["med_views"])} [{nr(fe["views_ci"][1])}–'
              f'{nr(fe["views_ci"][2])}], {fp(fe["mw_views_p"])})' if mv[0] == 'Fear' else
              f'{mv[0]} ({num(c.es(mv[0])["med_views"])}, n={c.es(mv[0])["n"]}) and Fear ({num(fe["med_views"])}, '
              f'{fp(fe["mw_views_p"])}) have the highest median views; Fear is the highest among labels with more than '
              f'100 videos')
             + f', but not more likes ({fp(fe["mw_likes_p"])}). '
             + (f'Qwen also puts Fear first ({num(R["eng_sent_q"]["Fear"]["med_views"])}).' if qmv == 'Fear' else
                f'Qwen {qs} puts {qmv} first ({num(R["eng_sent_q"][qmv]["med_views"])}).'))
    lowl = sorted(SE, key=lambda l: c.es(l)['med_likes'])[:2]
    h_, sa_ = c.es('Hope'), c.es('Sadness')
    A('Hope' in lowl)
    d.bullet(f'Videos labelled Hope and Sadness receive the lowest engagement — median likes of {num(h_["med_likes"])} and '
             f'{num(sa_["med_likes"])}, and median views of {num(h_["med_views"])} and {num(sa_["med_views"])} '
             f'(Hope likes {fp(h_["mw_likes_p"])}, views {fp(h_["mw_views_p"])}).')
    hu_ = c.es('Humour')
    msh = max(SE, key=lambda l: c.es(l)['mean_shares'])
    A(msh == 'Humour')
    d.bullet(f'Humour has the highest mean shares ({hu_["mean_shares"]:.1f}) and high median views '
             f'({num(hu_["med_views"])}), but its median likes and views do not differ significantly from other videos '
             f'({fp(hu_["mw_likes_p"])} and {fp(hu_["mw_views_p"])}).')
    di = c.es('Disgust')
    A(max(SE, key=lambda l: c.es(l)['mean_likes']) == 'Disgust')
    d.bullet(f'Disgust has the highest mean likes ({di["mean_likes"]:,.0f}) against a median of {num(di["med_likes"])}, '
             f'driven by a small number of highly viral videos.')
    d.h2('7.2  Engagement by Frame')
    d.fig(33, f'Median likes by frame with 95% bootstrap CIs — {mframe}.')
    d.fig(34, f'Median views by frame with 95% bootstrap CIs — {mframe}.')
    lf = sorted(FR11, key=lambda l: -c.ef(l)['med_likes'])
    vf = sorted(FR11, key=lambda l: -c.ef(l)['med_views'])
    if not FZ:
        p_ = c.ef('Personal connection')
        A(lf[0] == 'Personal connection' and max(FR11, key=lambda l: c.ef(l)['mean_shares']) == 'Personal connection')
        d.bullet(f'Personal connection content generates the highest median likes ({num(p_["med_likes"])}, '
                 f'{fp(p_["mw_likes_p"])}) and by far the highest mean shares ({p_["mean_shares"]:.1f}) and comments '
                 f'({p_["mean_comments"]:.1f}). However, this frame is over-predicted on the gold sample '
                 f'({P3("Personal connection")["npred"]} predictions against {P3("Personal connection")["support"]} gold '
                 f'rows) and its median shares are {num(p_["med_shares"])}, so these means reflect a few heavily shared '
                 f'videos; treat this finding as tentative.')
    else:
        h2_ = c.ef('Humanitarian')
        A(lf[0] == 'Humanitarian')
        d.bullet(f'Humanitarian content has the highest median likes ({num(h2_["med_likes"])}, {fp(h2_["mw_likes_p"])}), '
                 f'followed by Personal connection ({num(c.ef("Personal connection")["med_likes"])}, from '
                 f'{c.ef("Personal connection")["n"]} videos).')
        sm_ = c.ef('Satire/meme')
        A(max(FR11, key=lambda l: c.ef(l)['mean_shares']) == 'Satire/meme')
        d.bullet(f'Satire/meme has the highest mean shares ({sm_["mean_shares"]:.1f}) and mean comments '
                 f'({sm_["mean_comments"]:.1f}).')
    mi = c.ef('Military/strategic')
    A(vf[0] == 'Military/strategic' and mi['mw_views_p'] < 0.001)
    d.bullet(f'Military/strategic achieves the highest median views ({num(mi["med_views"])}, {fp(mi["mw_views_p"])}) — '
             f'military and conflict content is the most widely viewed frame. Military/strategic is under-detected '
             f'(recall {ms["r"]:.3f}), so the videos it captures may not be typical of the frame as a whole.')
    e_ = c.ef('Economic/sanctions')
    A(lf[-1] == 'Economic/sanctions' and e_['mw_likes_p'] < 0.001)
    A(min(FR11, key=lambda l: R['eng_frame_q'][l]['med_likes']) == 'Economic/sanctions')
    d.bullet(f'Economic/sanctions has the lowest median likes ({num(e_["med_likes"])}, {fp(e_["mw_likes_p"])}) in both '
             f'Mistral and Qwen, and below-average median views ({num(e_["med_views"])}, {fp(e_["mw_views_p"])}), despite '
             f'being one of the most common frames. Given its high F1 ({ecm["f1"]:.3f}), this is the most reliable '
             f'engagement finding — economic content generates below-average audience engagement.')
    rf = c.ef('Religious framing')
    A(vf[-1] == 'Religious framing')
    mvf = sorted(FR11, key=lambda l: -c.ef(l)['mean_views'])
    d.bullet(f'Religious framing has the lowest median views ({num(rf["med_views"])}) but the '
             f'{"second-" if mvf.index("Religious framing") == 1 else ""}highest mean views ({rf["mean_views"]:,.0f}'
             + (f', after {mvf[0]}' if mvf.index('Religious framing') == 1 else '')
             + ') — driven by a small number of extremely viral videos.')
    A(mvf.index('Religious framing') <= 1)
    d.h2('7.3  Shares and Comments')
    d.fig(35, 'Mean shares and mean comments by sentiment.')
    d.fig(36, 'Mean shares and mean comments by frame.')
    mcs = max(SE, key=lambda l: c.es(l)['mean_comments'])
    mcf = max(FR11, key=lambda l: c.ef(l)['mean_comments'])
    msf = max(FR11, key=lambda l: c.ef(l)['mean_shares'])
    d.para(f'Shares and comments follow the same pattern as likes only in part. By sentiment, '
           + (f'{msh} leads on both shares ({c.es(msh)["mean_shares"]:.1f}) and comments ({c.es(mcs)["mean_comments"]:.1f}).'
              if msh == mcs else f'{msh} has the highest mean shares ({c.es(msh)["mean_shares"]:.1f}) and {mcs} the '
                                 f'highest mean comments ({c.es(mcs)["mean_comments"]:.1f}).')
           + f' By frame, '
           + (f'{msf} leads on both shares ({c.ef(msf)["mean_shares"]:.1f}) and comments ({c.ef(mcf)["mean_comments"]:.1f}).'
              if msf == mcf else f'{msf} leads on shares ({c.ef(msf)["mean_shares"]:.1f}) and {mcf} on comments '
                                 f'({c.ef(mcf)["mean_comments"]:.1f}).')
           + f' Because these are means of very skewed counts, a '
           f'handful of viral videos decide the ranking; they are reported for completeness, not as findings.')

    # ── 8. Robustness ────────────────────────────────────────────────────────
    d.h1('8. Robustness Checks')
    d.h2('8.1  Across Prompt Settings')
    d.para('The same analysis was also run in two strict settings, where both tiers come from one prompt setting '
           '(Strict Zero-Shot and Strict Few-Shot reports). The table shows which findings survive a change of setting.'
           if S == 'best' else
           f'The same analysis was run for the best setting (Substantive report), strict zero-shot and strict few-shot. '
           f'Only the {"frame" if S == "zs" else "sentiment"} classifier differs between this report and the best-setting '
           f'report. The table shows which findings survive a change of setting.')
    d.table(['Finding', 'Best setting', 'Strict zero-shot', 'Strict few-shot'], robustness_rows(),
            [3000, 2000, 2000, 2000], size=18)
    d.caption('Table 5. Key findings under the three prompt settings (Mistral 7B unless stated).')
    d.h2('8.2  With Classification Error')
    d.para('Each main comparison was re-tested with the error-corrected shares from Section 1.5. The bootstrap resamples '
           'the gold rows and both groups together, so the interval for the difference includes classifier error. A '
           'finding is marked "Holds" when this interval excludes zero (p < 0.05) and "Not confirmed" otherwise. "Not '
           'confirmed" does not mean the difference is absent: for labels with only a few gold rows, the gold sample is '
           'too small to confirm or reject it.')
    rows = []
    for name, v in c.E['compare'].items():
        if name.startswith('Fear'):
            continue
        rows.append([name, f'{v["raw_a"]:.1f}% vs {v["raw_b"]:.1f}%', f'{v["corr_a"]:.1f}% vs {v["corr_b"]:.1f}%',
                     f'{v["diff"]:+.1f} [{v["diff_ci"][0]:+.1f}, {v["diff_ci"][1]:+.1f}]', pv(v['p']),
                     'Holds' if v['p'] < 0.05 else 'Not confirmed'])
    d.table(['Comparison', 'Classifier', 'Error-corrected', 'Difference [95% CI]', 'p', 'Result'], rows,
            [3000, 1400, 1400, 1700, 800, 1300], size=17)
    d.caption(f'Table 6. Main comparisons with classification error included — {msent} sentiment, {mframe} frames. '
              f'Week comparisons: first two weeks (March 2–15) against last two (April 20 – May 3).')
    held = [n for n, v in c.E['compare'].items() if v['p'] < 0.05]
    A(set(held) == {'Religious framing: Indonesian vs English', 'Religious framing: ID vs US', 'Religious framing: NG vs US',
                    'Humanitarian: first two weeks vs last two', 'Political blame: first two weeks vs last two'})
    d.para(f'Religious framing for Indonesian and Nigerian posts, the Political blame rise and the Humanitarian decline '
           f'hold. The corrected Religious framing shares are about half the classifier\'s, because Mistral also assigns '
           f'Religious framing to posts the annotators coded otherwise ({c.prec("Religious framing")[0]} of '
           f'{c.prec("Religious framing")[1]} predictions confirmed), but the gap between groups remains. The Solidarity '
           f'findings and the Historical context differences are not confirmed: Solidarity has 3 gold rows and Historical '
           f'context 3, so the correction is too uncertain. Fear comparisons cannot be tested at all, because the gold '
           f'sample contains no Fear.')

    # ── 9. Key findings ──────────────────────────────────────────────────────
    d.h1('9. Key Findings')
    ofr = c.E['overall_frame']
    d.bullet(f'Corrected for classifier error, Military/strategic is the most common frame '
             f'({ofr["Military/strategic"]["corrected"]:.1f}%) and Political blame is {ofr["Political blame"]["corrected"]:.1f}%, '
             f'in line with the human-coded sample. Mistral\'s raw output ({top_frame} {pc(c.d("Mistral", "frame", top_frame))}) '
             f'over-assigns Political blame.')
    d.bullet(f'Emotion is rare in these descriptions: annotators coded {R["gold_neutral_ci"][0]:.1f}% of the gold sample '
             f'Neutral, and no Fear at all (true share below {c.E["overall_sentiment"]["Fear"]["gold_exact"][1]:.1f}%). '
             f'Sentiment patterns in this report are patterns in the classifier\'s labels.')
    d.bullet('Weekly volume is shaped by the collection design (two-day windows plus month-end queries), so the apparent '
             'surges in March 30 – April 5 and the final week are not real changes in posting. As shares, Fear and Anger '
             'labels peak in April 6–12, mostly the April 6 window just before the ceasefire, in both Mistral and Qwen.')
    d.bullet(f'Political blame rises and Humanitarian framing falls between early March and late April, and both trends '
             f'hold with classification error included. The Solidarity decline ({pc(sol[0])} → {pc(sol[-1])}) is clear in '
             f'both classifiers but cannot be confirmed with the gold sample.')
    rc_ = c.ec('Religious framing: Indonesian vs English')
    d.bullet(f'Indonesian-language posts use Religious framing more often than English posts ({pc(id_rel["pa"])} vs '
             f'{pc(id_rel["pb"])}; corrected {pc(rc_["corr_a"])} vs {pc(rc_["corr_b"])}, {fp(rc_["p"])}), and Nigerian '
             f'posts more than US posts. These are differences in framing frequency; the data do not show why.')
    d.bullet(f'Humour → Satire/meme is the best-supported co-occurrence (lift {c.co("Humour", "Satire/meme")["lift"]:.0f}). '
             f'Anger → Political blame is the largest by share ({pc(ap["pct"])}) and holds in every setting, but Anger '
             f'predictions are rarely confirmed on the gold sample, so it describes the classifier\'s labels.')
    d.bullet(f'Economic/sanctions posts, the most reliably classified frame, receive the fewest likes in both models (median '
             f'{num(e_["med_likes"])}, {fp(e_["mw_likes_p"])}). Videos labelled Solidarity receive the most likes '
             f'(median {num(s_["med_likes"])}), but the Solidarity label itself is weakly validated.')
    d.bullet(f'High Unclear rates in non-English content mostly reflect the posts themselves: annotators coded '
             f'{ng["gold_unc_ne"]:.1f}% of non-English gold rows Unclear (vs {ng["gold_unc_en"]:.1f}% of English rows), and '
             f'Mistral\'s Unclear predictions on these rows matched the annotators {ng["prec_unc_ne"]:.1f}% of the time. '
             f'No cultural interpretation is drawn for German, Catalan or Pakistani content.')
    A(all(c.pt(k)['p'] > 0.05 for k in ('French|English|Fear', 'French|English|Anger', 'US|DE|Fear', 'AE|US|Anger')))
    d.bullet('Several comparisons from the earlier draft are not supported: French vs English Fear and Anger, US vs Germany '
             'Fear and UAE vs US Anger do not differ significantly, and British Solidarity and Indonesian '
             + ('and French ' if c.pt('French|English|Historical context')['p'] < 0.05 else '')
             + 'Historical context are not confirmed with classification error included.')

    # ── 10. Limitations ─────────────────────────────────────────────────────
    d.h1('10. Limitations')
    d.bullet(f'The gold standard has {n_neu} Neutral rows out of 400. Emotional labels have 0–10 gold rows each, and Fear, '
             f'Sadness and Misinformation have none, so their reliability cannot be measured.')
    d.bullet('Classification error is carried into the overall shares and the main comparisons (Sections 1.5 and 8.2). '
             'Other percentages in Sections 3–7 carry sampling uncertainty only. The correction assumes the classifier '
             'makes the same kinds of errors in every group and week; with 87 non-English gold rows this cannot be checked '
             'for individual non-English communities.')
    d.bullet('The sample is nine two-day windows plus month-end bulk queries, not a continuous record. Weekly results '
             'describe those windows, and events between windows are not observed.')
    d.bullet(f'Posts were classified from English translations, and annotators also read the translations first. Frame '
             f'assignment on non-English posts is weaker on the gold sample ({ng["n_framed_ne"]} rows), and German, Catalan '
             f'and Pakistani findings rest largely on short or hashtag-only posts.')
    if not SZ:
        d.bullet('Mistral few-shot received a different Fear example from Qwen few-shot, so Fear rates are not strictly '
                 'comparable between the two models.')
    if S == 'zs':
        d.bullet(f'Qwen zero-shot sentiment agrees with the gold standard at near-chance level (Kappa '
                 f'{ev["Qwen"]["t1"]["kappa"]:.3f}), which limits it as a cross-check for sentiment.')
    d.bullet('The analysis covers video descriptions only — not video content, audio, or visual elements.')
    d.bullet('DeepSeek-R1 classification is partial (2,700–3,000 of 5,848 videos). Language, regional, and time-series '
             'analyses for DeepSeek are not reported to avoid systematic bias from incomplete coverage.')
    d.bullet('TikTok\'s algorithmic recommendation and the API\'s search results shape the dataset — geographic and '
             'linguistic distribution reflects curation as much as organic public interest.')
    d.bullet('Persian and Arabic communities are below 1% of the dataset — too small for reliable analysis despite being '
             'primary stakeholder communities in the Iran crisis.')

    # ── Appendix ─────────────────────────────────────────────────────────────
    d.h1('Appendix A. Label Shares')
    rows = [[l, ci(R['ci_sent'][l]), pc(c.d('Qwen', 'sent', l)), pc(c.d('DeepSeek', 'sent', l))] for l in SENT9 + ['Neutral', 'Unclear']]
    d.table(['Sentiment', f'Mistral {ss} [95% CI]', f'Qwen {qs}', 'DeepSeek (partial)'], rows, [2200, 2900, 1900, 2000], size=18)
    d.caption('Table A1. Sentiment shares (% of videos).')
    rows = [[l, ci(R['ci_frame'][l]), pc(c.d('Qwen', 'frame', l)), pc(c.d('DeepSeek', 'frame', l)), ci(gfc[l])] for l in FR11]
    d.table(['Frame', f'Mistral {fs_} [95% CI]', f'Qwen {qs}', 'DeepSeek', 'Gold sample [95% CI]'], rows,
            [2300, 2300, 1300, 1300, 2300], size=18)
    d.caption('Table A2. Frame shares (% of videos; gold sample = % of the 400 annotated rows).')

    d.h1('Appendix B. Weekly Shares')
    for tag, labs, nm in (('sent_m', ['Fear', 'Anger', 'Solidarity', 'Humour'], 'sentiment'),
                          ('frame_m', ['Political blame', 'Military/strategic', 'Economic/sanctions', 'Humanitarian'], 'frame')):
        rows = []
        for i, w in enumerate(c.wk):
            rows.append([WK[i], f'{wt[w]:,}', f'{wtn[w]:,}']
                        + [f'{R["weekly"][tag][l][w]:.1f} ({R["weekly"][tag + "_nomonth"][l][w]:.1f})' for l in labs])
        d.table(['Week', 'Videos', 'No month q.'] + labs, rows, [1900, 900, 1000] + [1300] * 4, size=17)
        d.caption(f'Table B{1 if nm == "sentiment" else 2}. Weekly {nm} shares (%) — {M}. In brackets: without '
                  f'month-query videos.')

    d.h1('Appendix C. Pairwise Comparisons')
    rows = []
    for k, v in R['pair_tests'].items():
        a, b, lab = k.split('|')
        rows.append([lab, f'{a} vs {b}', f'{v["pa"]:.1f}% ({v["a"]}/{v["na"]:,})', f'{v["pb"]:.1f}% ({v["b"]}/{v["nb"]:,})',
                     pv(v['p'])])
    d.table(['Label', 'Comparison', 'Group A', 'Group B', 'Fisher p'], rows, [2100, 2200, 1700, 1800, 1200], size=17)
    d.caption('Table C1. Fisher\'s exact tests for the community comparisons discussed in Sections 5 and 6.')

    d.h1('Appendix D. Engagement')
    for key, labs, nm in (('eng_sent_m', SE, 'sentiment'), ('eng_frame_m', FR11, 'frame')):
        rows = []
        for l in labs:
            v = R[key][l]
            rows.append([l, f'{v["n"]:,}', f'{num(v["med_likes"])} [{nr(v["likes_ci"][1])}–{nr(v["likes_ci"][2])}]',
                         pv(v['mw_likes_p']),
                         f'{num(v["med_views"])} [{nr(v["views_ci"][1])}–{nr(v["views_ci"][2])}]',
                         pv(v['mw_views_p']), f'{v["mean_shares"]:.1f}', f'{v["mean_comments"]:.1f}'])
        d.table([nm.capitalize(), 'n', 'Median likes [CI]', 'p', 'Median views [CI]', 'p', 'Mean shares', 'Mean comm.'],
                rows, [1900, 750, 1500, 900, 1600, 900, 900, 900], size=16)
        d.caption(f'Table D{1 if nm == "sentiment" else 2}. Engagement by {nm} — {M}. p = Mann–Whitney U against all other '
                  f'labelled videos.')

    d.h1('Appendix E. Data Notes')
    d.bullet(f'{R["n_no_date"]} videos have no timestamp; 4 videos have a damaged date_range field and are counted with '
             f'the window queries.')
    d.bullet('Gold rows were matched to classified videos on the first 18 digits of the video id, as in the evaluation '
             'scripts. "Humor" labels in the classifier output were renamed "Humour".')
    d.bullet('The week of February 23 – March 1 holds only 8 videos (from March 1) and is left out of the weekly analysis.')
    d.bullet(f'The {R["short"]["n"]} short descriptions (under 10 characters or 4 words) were labelled Unclear without a '
             f'model call in the few-shot runs only'
             + (f'; in this report they affect the frame labels only.' if S == 'best' else
                f'; this report uses zero-shot labels, so they were classified normally.' if S == 'zs' else
                f'; they make up most of the {R["counts"]["sent"]["Unclear"]} Unclear sentiment labels.'))
    d.bullet('Region codes with fewer than 50 videos are not compared, except where Malaysia is mentioned as a check on '
             'the Indonesian result.')


def robustness_rows():
    out = []
    C = {s: Ctx(s) for s in ('best', 'zs', 'fs')}

    def row(name, fn):
        out.append([name] + [fn(C[s]) for s in ('best', 'zs', 'fs')])

    row('Neutral share', lambda c: pc(c.d('Mistral', 'sent', 'Neutral')))
    row('Most frequent emotional label', lambda c: (lambda l: f'{l} ({pc(c.d("Mistral", "sent", l))})')(
        max(SENT9, key=lambda l: c.d('Mistral', 'sent', l))))
    row('Political blame share (gold 6.8%)', lambda c: pc(c.d('Mistral', 'frame', 'Political blame')))
    row('Military/strategic share (gold 34.2%)', lambda c: pc(c.d('Mistral', 'frame', 'Military/strategic')))
    row('Anger → Political blame', lambda c: f'{pc(c.co("Anger", "Political blame")["pct"])} (lift '
                                             f'{c.co("Anger", "Political blame")["lift"]:.1f})')
    row('Fear → top frame', lambda c: f'{c.cotop("Fear")} ({pc(c.co("Fear", c.cotop("Fear"))["pct"])})')
    row('Humour → Satire/meme', lambda c: f'{pc(c.co("Humour", "Satire/meme")["pct"])} (lift '
                                          f'{c.co("Humour", "Satire/meme")["lift"]:.0f})')
    row('Sadness → Humanitarian', lambda c: pc(c.co('Sadness', 'Humanitarian')['pct']))
    row('Solidarity trend (z)', lambda c: f'{c.tt("Solidarity_m")["z"]:.1f} ({fp(c.tt("Solidarity_m")["p"])})')
    row('Humanitarian trend (z)', lambda c: f'{c.tt("Humanitarian_m")["z"]:.1f} ({fp(c.tt("Humanitarian_m")["p"])})')
    row('Religious framing, Indonesian vs English', lambda c: (lambda v: f'{v["pa"]:.1f}% vs {v["pb"]:.1f}% '
                                                                         f'({fp(v["p"])})')(c.pt('Indonesian|English|Religious framing')))
    row('Solidarity, GB vs US', lambda c: (lambda v: f'{v["pa"]:.1f}% vs {v["pb"]:.1f}% ({fp(v["p"])})')(c.pt('GB|US|Solidarity')))
    row('Highest Fear (8 regions)', lambda c: (lambda g: f'{g} ({pc(c.sh("reg", "sent", g, "Fear"))})')(c.top('reg', 'sent', 'Fear')))
    row('Top median likes, sentiment', lambda c: (lambda l: f'{l} ({num(c.es(l)["med_likes"])})')(
        max(SENT9 + ['Neutral'], key=lambda l: c.es(l)['med_likes'])))
    row('Top median views, sentiment (n ≥ 100)', lambda c: (lambda l: f'{l} ({num(c.es(l)["med_views"])})')(
        max([l for l in SENT9 + ['Neutral'] if c.es(l)['n'] >= 100], key=lambda l: c.es(l)['med_views'])))
    row('Top median likes, frame', lambda c: (lambda l: f'{l} ({num(c.ef(l)["med_likes"])})')(
        max(FR11, key=lambda l: c.ef(l)['med_likes'])))
    row('Lowest median likes, frame', lambda c: (lambda l: f'{l} ({num(c.ef(l)["med_likes"])})')(
        min(FR11, key=lambda l: c.ef(l)['med_likes'])))
    row('Top median views, frame', lambda c: (lambda l: f'{l} ({num(c.ef(l)["med_views"])})')(
        max(FR11, key=lambda l: c.ef(l)['med_views'])))
    return out
