# Final combined report: the best setting is the main analysis; the strict zero-shot and few-shot settings are
# robustness checks. Every number is read from data/analysis/report_<s>/results.json,
# data/analysis/report_<s>/classification_error.json and data/analysis/final/fewshot_overlap.json; ranking and
# significance claims are checked with assertions before the text is written.
# Run first: analysis_core.py, classification_error.py and report_figs.py for best, zs and fs; final_figs.py;
# fewshot_overlap.py.
# Usage (project root): python 07_reports/final_report.py <template.docx> <out.docx>
# template.docx: any earlier report .docx; only its page setup, styles and numbering are used.
import sys, json
import numpy as np
from docx import Document
from docx.shared import Inches, Emu
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.text.paragraph import Paragraph
from xml.sax.saxutils import escape
from PIL import Image
import pandas as pd

from report_text import (Ctx, RES, A, pc, ci, fp, pv, nr, num, lst, times, WK, REG8, LANG6, REGNAME, SENT9, FR10, FR11,
                         robustness_rows)

TEMPLATE, OUT = sys.argv[1:3]
doc = Document(TEMPLATE)
body = doc.element.body
for el in list(body):
    if el.tag != qn('w:sectPr'):
        body.remove(el)
SECT = body.find(qn('w:sectPr'))
sec = doc.sections[0]
TEXT_W = sec.page_width - sec.left_margin - sec.right_margin
MAX_W, MAX_H = min(Inches(6.3), TEXT_W), Emu(int(16 * 360000))       # figures never wider than the text, <= 16 cm tall
FONT = '<w:rFonts w:ascii="Calibri" w:cs="Calibri" w:eastAsia="Calibri" w:hAnsi="Calibri"/>'
BR = 'data/analysis/report_best'
FX = 'data/analysis/final'


# ── document helpers (no empty spacer paragraphs; keep-with-next only on headings and figure images) ──────────
def add(xml):
    el = parse_xml(xml)
    SECT.addprevious(el)
    return el


def run(text, bold=False, italic=False, size=22, color='000000'):
    b = '<w:b/><w:bCs/>' if bold else '<w:b w:val="false"/><w:bCs w:val="false"/>'
    i = '<w:i/><w:iCs/>' if italic else '<w:i w:val="false"/><w:iCs w:val="false"/>'
    return (f'<w:r><w:rPr>{FONT}{b}{i}<w:color w:val="{color}"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
            f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r>')


def title(t, sub, sub2):
    add(f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:after="60" w:before="0"/><w:jc w:val="center"/></w:pPr>{run(t, True, size=40)}</w:p>')
    add(f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:after="40" w:before="0"/><w:jc w:val="center"/></w:pPr>'
        f'{run(sub, italic=True, color="555555")}</w:p>')
    add(f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:after="240" w:before="0"/><w:jc w:val="center"/></w:pPr>'
        f'{run(sub2, color="555555", size=20)}</w:p>')


def h1(t, page_break=False):
    pb = '<w:pageBreakBefore/>' if page_break else ''
    add(f'<w:p {nsdecls("w")}><w:pPr><w:pStyle w:val="Heading1"/><w:keepNext/>{pb}<w:spacing w:after="120" w:before="320"/></w:pPr>'
        f'{run(t, True, size=28)}</w:p>')


def h2(t):
    add(f'<w:p {nsdecls("w")}><w:pPr><w:pStyle w:val="Heading2"/><w:keepNext/><w:spacing w:after="80" w:before="240"/></w:pPr>'
        f'{run(t, True, size=24)}</w:p>')


def para(text, lead=None, italic=False):
    r = (run(lead, True) if lead else '') + run(text.strip(), italic=italic)
    add(f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:after="60" w:before="60"/></w:pPr>{r}</w:p>')


def bullet(text, lead=None):
    r = (run(lead, True) if lead else '') + run(text)
    add(f'<w:p {nsdecls("w")}><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>'
        f'<w:spacing w:after="60" w:before="60"/></w:pPr>{r}</w:p>')


FIGS = []


def fig(path, caption, label=None, width=6.3):
    w, h = Image.open(path).size
    W = min(Inches(width), MAX_W)
    if W * h / w > MAX_H:
        W = Emu(int(MAX_H * w / h))
    p = add(f'<w:p {nsdecls("w")}><w:pPr><w:keepNext/><w:spacing w:after="60" w:before="160"/><w:jc w:val="center"/></w:pPr></w:p>')
    Paragraph(p, doc._body).add_run().add_picture(path, width=W)
    FIGS.append(path)
    if label is None:
        FN[0] += 1
    lab = label or f'Figure {FN[0]}'
    add(f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:after="160" w:before="0"/><w:jc w:val="center"/></w:pPr>'
        f'{run(f"{lab}. {caption}", italic=True, size=19, color="555555")}</w:p>')


FN = [0]


def cell(text, w, header=False, align='center', size=22, bold=False, keep=False):
    fill = 'F2F2F2' if header else 'FFFFFF'
    kn = '<w:keepNext/>' if keep else ''
    m = 80 if header else 70
    return (f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/><w:tcBorders>'
            + ''.join(f'<w:{s} w:val="single" w:color="000000" w:sz="4"/>' for s in ('top', 'left', 'bottom', 'right'))
            + f'</w:tcBorders><w:shd w:fill="{fill}" w:val="clear"/><w:tcMar><w:top w:type="dxa" w:w="{m}"/>'
            f'<w:left w:type="dxa" w:w="120"/><w:bottom w:type="dxa" w:w="{m}"/><w:right w:type="dxa" w:w="120"/></w:tcMar></w:tcPr>'
            f'<w:p><w:pPr>{kn}<w:spacing w:after="0" w:before="0"/><w:jc w:val="{align}"/></w:pPr>{run(text, bold=header or bold, size=size)}</w:p></w:tc>')


def table(headers, rows, widths, size=20, bold_rows=()):
    A(sum(widths) <= TEXT_W / 635, 'table wider than the text area')                 # 635 EMU per twip
    g = ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    # short tables stay on one page; every table keeps its last row with the caption that follows it
    whole = len(rows) <= 20
    hdr = ('<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>'
           + ''.join(cell(h, w, True, 'left' if i == 0 else 'center', size, keep=True) for i, (h, w) in enumerate(zip(headers, widths)))
           + '</w:tr>')
    rws = ''.join('<w:tr><w:trPr><w:cantSplit/></w:trPr>'
                  + ''.join(cell(v, w, False, 'left' if i == 0 else 'center', size, bold=k in bold_rows,
                                 keep=whole or k >= len(rows) - 2)
                            for i, (v, w) in enumerate(zip(r, widths))) + '</w:tr>' for k, r in enumerate(rows))
    add(f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:type="dxa" w:w="{sum(widths)}"/><w:tblBorders>'
        + ''.join(f'<w:{s} w:val="single" w:color="auto" w:sz="4"/>' for s in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'))
        + f'</w:tblBorders><w:jc w:val="center"/></w:tblPr><w:tblGrid>{g}</w:tblGrid>{hdr}{rws}</w:tbl>')


def caption(text):
    add(f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:after="160" w:before="60"/><w:jc w:val="center"/></w:pPr>'
        f'{run(text, italic=True, size=19, color="555555")}</w:p>')


# ── numbers ──────────────────────────────────────────────────────────────────
c = Ctx('best')
R, E = c.R, c.E
EZ, EF = RES['zs']['eval'], RES['fs']['eval']
OV = json.load(open(f'{FX}/fewshot_overlap.json'))
coll = R['collection']
gfc = R['gold_frame_ci']
osent, ofr = E['overall_sentiment'], E['overall_frame']
P1 = lambda l: c.pcl('Mistral', 't1', l)
P3 = lambda l: c.pcl('Mistral', 't3', l)
emo_rank = sorted(SENT9, key=lambda l: -c.d('Mistral', 'sent', l))
top_frame = max(FR10, key=lambda l: c.d('Mistral', 'frame', l))
rel_c = c.ec('Religious framing: Indonesian vs English')
ngr_c, idr_c = c.ec('Religious framing: NG vs US'), c.ec('Religious framing: ID vs US')
pbw_c, huw_c = c.ec('Political blame: first two weeks vs last two'), c.ec('Humanitarian: first two weeks vs last two')
solw_c, gbs_c = c.ec('Solidarity: first two weeks vs last two'), c.ec('Solidarity: GB vs US')
hci_c, hcf_c = c.ec('Historical context: Indonesian vs English'), c.ec('Historical context: French vs English')
A(rel_c['p'] < 0.05 and ngr_c['p'] < 0.05 and idr_c['p'] < 0.05 and pbw_c['p'] < 0.05 and huw_c['p'] < 0.05)
A(solw_c['p'] > 0.05 and gbs_c['p'] > 0.05 and hci_c['p'] > 0.05 and hcf_c['p'] > 0.05)
fear_up = osent['Fear']['gold_exact'][1]
n_neu = R['gold_counts']['t1']['Neutral']
id_rel = c.pt('Indonesian|English|Religious framing')
ng_rel = c.pt('NG|US|Religious framing')
gb_sol = c.pt('GB|US|Solidarity')
ng = R['ne_gold']
neo = R['ne_overall']
ne = R['nonenglish']
k_ec, n_ec = c.prec('Economic/sanctions')
e_, s_ = c.ef('Economic/sanctions'), c.es('Solidarity')
mi = c.ef('Military/strategic')
ms, pbm = P3('Military/strategic'), P3('Political blame')
ecm, hu = P3('Economic/sanctions'), P1('Humour')
W = c.W
sol, solq = W('sent_m', 'Solidarity'), W('sent_q', 'Solidarity')
pbw, huw, huq = W('frame_m', 'Political blame'), W('frame_m', 'Humanitarian'), W('frame_q', 'Humanitarian')
fpk, fpv = c.peak('sent_m', 'Fear')
apk, apv = c.peak('sent_m', 'Anger')
A(fpk == WK[5] and apk == WK[5])
sol_t, hum_t, pol_t, fear_t = c.tt('Solidarity_m'), c.tt('Humanitarian_m'), c.tt('Political_m'), c.tt('Fear_m')
A(sol_t['p'] < 0.001 and hum_t['p'] < 0.001 and pol_t['p'] < 0.001 and fear_t['p'] > 0.05)
A(c.tt('Solidarity_m_nomonth')['p'] < 0.001 and c.tt('Humanitarian_m_nomonth')['p'] < 0.001 and c.tt('Political_m_nomonth')['p'] < 0.001)
ap = c.co('Anger', 'Political blame')
hs = c.co('Humour', 'Satire/meme')
wt, wtn = R['week_totals'], R['week_totals_nomonth']
br = coll['by_range']
daily = R['daily']
dsn = R['n_deepseek_sent'], R['n_deepseek_frame']
fs_ov = OV['scores']
pr = {l: c.prec(l) for l in ('Anger', 'Fear', 'Solidarity', 'Humour', 'Religious framing')}

# DeepSeek zero-shot and few-shot cover different gold rows (173 and 213); compare them on the rows both cover
import re as _re
from sklearn.metrics import cohen_kappa_score as _K
_nid = lambda x: _re.sub(r'[^0-9]', '', str(x))[:18]
_g = pd.read_csv('data/clean/gold_standard_dataset.csv')
_g['k'] = _g['id'].map(_nid)
_ds = {}
for _f in ('deepseek', 'deepseek_fewshot'):
    _d = pd.read_csv(f'data/clean/iran_crisis_videos_classified_{_f}.csv', low_memory=False, dtype={'id': str})
    _d['k'] = _d['id'].map(_nid)
    _ds[_f] = _d.drop_duplicates('k').set_index('k')['frame_label'].astype(str)
_common = _g[_g['k'].isin(_ds['deepseek'].index) & _g['k'].isin(_ds['deepseek_fewshot'].index)]
DS_COMMON = {f: _K(_common['tier3_frame'], _common['k'].map(_ds[f])) for f in _ds}
DS_N = len(_common)
A(DS_N == EZ['DeepSeek']['n_t3'] and DS_COMMON['deepseek_fewshot'] > DS_COMMON['deepseek'])
# Model ranking claims used in the text
A(EZ['Mistral']['t1']['kappa'] > max(EF['Mistral']['t1']['kappa'], EZ['Qwen']['t1']['kappa'], EF['Qwen']['t1']['kappa']))
A(all(EF[m]['t3']['kappa'] > EZ[m]['t3']['kappa'] for m in ('Mistral', 'Qwen', 'DeepSeek')))
A(EF['Qwen']['t3']['kappa'] > EF['Mistral']['t3']['kappa'])
A(max(v['t1']['kappa'] for v in list(EZ.values()) + list(EF.values())) < 0.2)
A(EZ['Mistral']['t1']['kappa'] > EZ['DeepSeek']['t1']['kappa'] and EZ['DeepSeek']['t1']['kappa'] > EF['DeepSeek']['t1']['kappa'])
A(EF['Qwen']['t1']['kappa'] > EZ['Qwen']['t1']['kappa'])
A(EF['Qwen']['t3']['kappa'] == max(EF[m]['t3']['kappa'] for m in ('Mistral', 'Qwen')))
A(R['gold_counts']['t1']['Solidarity'] == 3 and R['gold_counts']['t3']['Historical context'] == 3)
A(sorted(FR11, key=lambda l: -gfc[l][0])[:3] == ['Military/strategic', 'Unclear', 'Economic/sanctions'])

# ═════════════════════════════════════════════════════════════════════════════
title('Public Sentiment and Narrative Framing on TikTok during the 2026 Iran Crisis',
      'Final Report  |  5,848 TikTok video descriptions, March–April 2026',
      'Gourav Kumawat  |  Research internship, LMU Munich  |  Supervisor: Dr. Anna-Carolina Haensch  |  October 2026')

# ── Summary ──────────────────────────────────────────────────────────────────
h1('Summary')
para('This report brings together the evaluation and the analysis phases of the project. Open-source language models '
     'classified 5,848 TikTok video descriptions about the 2026 Iran crisis by sentiment and by narrative frame. A '
     '400-row gold standard was used to measure how well each model works and to correct the main estimates for '
     'classification error. The main results:')
bullet(f'Frame classification works moderately well and sentiment classification does not. Against the gold standard, '
       f'Tier 3 (frame) κ is {min(v["t3"]["kappa"] for v in list(EZ.values()) + list(EF.values()) if "t3" in v):.2f}–'
       f'{max(v["t3"]["kappa"] for v in list(EZ.values()) + list(EF.values()) if "t3" in v):.2f} across models and '
       f'settings, while Tier 1 (sentiment) κ is at most '
       f'{max(v["t1"]["kappa"] for v in list(EZ.values()) + list(EF.values())):.2f}. Few-shot prompting improves frames '
       f'for every model.', 'Models: ')
bullet(f'Most posts carry no clear emotion: the gold standard is {R["gold_neutral_ci"][0]:.1f}% Neutral and contains no '
       f'Fear (true share below {fear_up:.1f}%). The models give emotional labels far more often, so sentiment results '
       f'describe the classifier\'s labels, not confirmed emotion.', 'Sentiment: ')
bullet(f'Corrected for classification error, Military/strategic is the most common frame '
       f'({ofr["Military/strategic"]["corrected"]:.1f}%), followed by Economic/sanctions '
       f'({ofr["Economic/sanctions"]["corrected"]:.1f}%) and Political blame ({ofr["Political blame"]["corrected"]:.1f}%). '
       f'Mistral\'s raw output puts {top_frame} first ({pc(c.d("Mistral", "frame", top_frame))}) because it over-assigns '
       f'it.', 'Frames: ')
bullet(f'Political blame rises and Humanitarian framing falls from early March to late April, and both trends hold with '
       f'classification error included. Weekly video counts reflect the collection design, not posting activity.',
       'Over time: ')
bullet(f'Indonesian-language posts use Religious framing more often than English posts (corrected '
       f'{pc(rel_c["corr_a"])} vs {pc(rel_c["corr_b"])}, {fp(rel_c["p"])}), and Nigerian posts more than US posts. '
       f'Other community differences in the classifier output (British Solidarity, Indonesian and French Historical '
       f'context) are not confirmed once classification error is included.', 'Communities: ')
bullet(f'Economic/sanctions posts, the most reliably classified frame, receive the fewest likes (median '
       f'{num(e_["med_likes"])}, {fp(e_["mw_likes_p"])}). Military/strategic posts receive the most views.',
       'Engagement: ')

# ── 1. Project and data ──────────────────────────────────────────────────────
h1('1. Project and Data')
para('The project had three questions:')
bullet('How well do open-source language models classify sentiment and narrative frames in short TikTok descriptions, '
       'measured against a human-coded gold standard?')
bullet('Which sentiments and frames appear in TikTok posts about the 2026 Iran crisis, and how do they change over '
       'March and April?')
bullet('How do they differ across language and region communities, and how do they relate to engagement?')
h2('1.1  Data Collection and Preprocessing')
bullet(f'TikTok Research API, 18 Iran-related hashtags (the largest are #OilPrices, #IranConflict, #StraitOfHormuz, '
       f'#IranWar and #Iran). Video descriptions and their metadata (date, region, language, likes, views, shares, '
       f'comments) were collected.', 'Source: ')
bullet(f'Nine two-day query windows, each a Sunday and Monday from 1–2 March to 26–27 April '
       f'({coll["window_n"]:,} videos), plus one query for each whole month ({coll["month_query_n"]:,} videos). The '
       f'month queries returned mostly videos from the last days of each month.', 'Collection: ')
bullet('URLs, emojis, hashtags and all non-ASCII characters were removed from the text. Duplicates (same video id or same text), descriptions '
       'under 3 words and spam patterns (for example "link in bio") were dropped. Language was detected with langdetect, '
       'and non-English descriptions were translated with Helsinki-NLP opus-mt-mul-en.', 'Cleaning: ')
bullet(f'5,848 videos. {R["lang_n"]["English"]:,} ({R["lang_n"]["English"] / 58.48:.1f}%) are in English; the next '
       f'largest languages are Indonesian ({R["lang_n"]["Indonesian"]}), German ({R["lang_n"]["German"]}), Spanish '
       f'({R["lang_n"]["Spanish"]}), French ({R["lang_n"]["French"]}) and Catalan ({R["lang_n"]["Catalan"]}). The US '
       f'({R["reg_n"]["US"]:,}) and the UK ({R["reg_n"]["GB"]}) are the largest regions.', 'Final dataset: ')
para(f'Weekly video counts therefore reflect the collection, not how much was posted. The two month queries '
     f'({br["20260301_20260331"]} videos for March, {br["20260401_20260430"]} for April) put '
     f'{daily["2026-03-31"]["month"]} videos on March 31 and {daily["2026-04-30"]["month"]} on April 30. All trends in '
     f'this report are weekly shares, and each trend was re-tested without the month-query videos.')
para('The war began on February 28 with US and Israeli strikes on Iran, one day before the data start. Events inside '
     'the period, marked in Figure 1: on March 25 Iran rejected a US 15-point proposal passed on by Pakistan; on April 7 '
     'a two-week ceasefire was announced; on April 12 the Islamabad talks collapsed and the US announced a naval '
     'blockade from April 13; on April 21 the ceasefire was extended while the blockade stayed in place. Because each '
     'week is mostly one Sunday–Monday window, a weekly share describes those two days, not the whole week.')
fig(f'{BR}/fig08.png', 'Videos per day by collection query, with events. Two-day window queries in blue, month queries in red.')

h2('1.2  Coding Scheme and Gold Standard')
para(f'Each post gets one label on two tiers. Tier 1 (sentiment) has nine emotions plus Neutral. Tier 3 (narrative '
     f'frame) has ten frames plus Unclear, for posts that are off-topic, too short or fit no frame. The label '
     f'definitions are in Appendix A.')
para(f'The gold standard is a simple random sample of 400 of the 5,848 videos. Two coders, Gourav Kumawat and '
     f'Anna-Carolina Haensch, labelled all 400 rows independently, following the annotation guidelines. Agreement was '
     f'κ = 0.85 for Tier 1 (98.2% raw agreement) and κ = 0.89 for Tier 3 (91.5%). The remaining disagreements (7 on '
     f'Tier 1, 34 on Tier 3) were discussed and resolved to give the final labels.')
para(f'In the gold standard, {n_neu} of 400 rows ({R["gold_neutral_ci"][0]:.1f}%) are Neutral and the emotional labels '
     f'have 0–10 rows each; Fear, Sadness and Misinformation have none. The most common frames are Military/strategic '
     f'({ci(gfc["Military/strategic"])}), Unclear ({ci(gfc["Unclear"])}) and Economic/sanctions '
     f'({ci(gfc["Economic/sanctions"])}). Because the sample is random, these shares are also direct estimates for the '
     f'whole dataset.')

# ── 2. Classification and evaluation ─────────────────────────────────────────
h1('2. Classification and Model Evaluation')
h2('2.1  Models and Prompt Settings')
bullet('Mistral-7B-Instruct-v0.3, Qwen2.5-3B-Instruct and DeepSeek-R1-0528-Qwen3-8B classified both tiers. '
       'Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) gave a three-class sentiment baseline '
       '(positive, negative, neutral).', 'Models: ')
bullet('Sentiment and frame were classified in two separate prompts, each asking for exactly one label from a list with '
       'short definitions (Appendix A).', 'Prompts: ')
bullet(f'The few-shot prompts add two labelled examples per label ({OV["n_examples"]} examples in total). In the few-shot runs, descriptions shorter than 10 characters or 4 words were set '
       f'to Unclear without calling the model ({R["short"]["n"]} videos).', 'Zero-shot and few-shot: ')
bullet(f'Every post was classified from the English translation of its description.', 'Input: ')
bullet(f'DeepSeek-R1 is partial: {dsn[0]:,} videos zero-shot ({EZ["DeepSeek"]["n_t1"]} gold rows) and {dsn[1]:,} '
       f'few-shot ({EF["DeepSeek"]["n_t1"]} gold rows). Its scores are indicative only, and it is not used in the '
       f'language, region or time analyses.', 'DeepSeek: ')

h2('2.2  Benchmark')
para('Table 1 compares every model and setting with the gold standard. Cohen\'s κ corrects for chance agreement. Macro '
     'F1 weights every label equally, so it shows how well the rare labels are classified; weighted F1 and raw agreement '
     'are dominated by the Neutral rows and are given in Appendix C.')
T = lambda E_, m, t, k: f'{E_[m][t][k]:.3f}' if m in E_ and t in E_[m] else '—'
used = {('Mistral', 'zs', 't1'), ('Mistral', 'fs', 't3'), ('Qwen', 'fs', 't1'), ('Qwen', 'fs', 't3'),
        ('DeepSeek', 'zs', 't1'), ('DeepSeek', 'fs', 't3'), ('RoBERTa', 'zs', 't1')}
rows = []
for m, nm in (('Mistral', 'Mistral 7B'), ('Qwen', 'Qwen2.5-3B'), ('DeepSeek', 'DeepSeek-R1 (partial)'), ('RoBERTa', 'RoBERTa (3-class)')):
    for s_tag, E_, sname in (('zs', EZ, 'Zero-shot'), ('fs', EF, 'Few-shot')):
        if m not in E_:
            continue
        mk = lambda t: ' *' if (m, s_tag, t) in used and m != 'RoBERTa' else ''
        gold_n = str(E_[m]['n_t1'])
        rows.append([nm, sname, T(E_, m, 't1', 'kappa') + mk('t1'), T(E_, m, 't1', 'mf1'),
                     T(E_, m, 't3', 'kappa') + mk('t3'), T(E_, m, 't3', 'mf1'), gold_n])
table(['Model', 'Setting', 'Tier 1 κ', 'Tier 1 macro F1', 'Tier 3 κ', 'Tier 3 macro F1', 'Gold rows'], rows,
      [2300, 1150, 1050, 1250, 1050, 1250, 950], size=19)
caption('Table 1. Agreement with the gold standard for every model and prompt setting. * = setting used in the analysis '
        '(best setting per model and tier). RoBERTa is scored on three classes and has no frame or few-shot setting.')
fig(f'{FX}/bench.png', 'Cohen\'s κ against the gold standard by model and prompt setting. Hatched bars: DeepSeek-R1, '
    f'scored on {EZ["DeepSeek"]["n_t1"]} (zero-shot) and {EF["DeepSeek"]["n_t1"]} (few-shot) gold rows.')
para(f'Sentiment agreement is low for every model (κ ≤ {max(v["t1"]["kappa"] for v in list(EZ.values()) + list(EF.values())):.2f}). '
     f'The gold standard is almost all Neutral, and every model gives emotional labels to many of those posts, so a '
     f'handful of correct emotional labels is outweighed by many false ones. Frame agreement is moderate: few-shot '
     f'prompting raises Tier 3 κ for all three models (Mistral {EZ["Mistral"]["t3"]["kappa"]:.2f} → '
     f'{EF["Mistral"]["t3"]["kappa"]:.2f}, Qwen {EZ["Qwen"]["t3"]["kappa"]:.2f} → {EF["Qwen"]["t3"]["kappa"]:.2f}, '
     f'DeepSeek {DS_COMMON["deepseek"]:.2f} → {DS_COMMON["deepseek_fewshot"]:.2f} on the {DS_N} gold rows both of its runs '
     f'cover). For sentiment, few-shot '
     f'helps Qwen but lowers Mistral ({EZ["Mistral"]["t1"]["kappa"]:.2f} → {EF["Mistral"]["t1"]["kappa"]:.2f}) and '
     f'DeepSeek.')
para(f'The analysis therefore uses the best setting per model and tier: Mistral zero-shot for sentiment and few-shot for '
     f'frames, Qwen few-shot for both, and DeepSeek zero-shot for sentiment and few-shot for frames. Mistral is the '
     f'primary model, because it has the highest Tier 1 κ and full coverage. Qwen few-shot has a higher Tier 3 κ than '
     f'Mistral ({EF["Qwen"]["t3"]["kappa"]:.2f} vs {EF["Mistral"]["t3"]["kappa"]:.2f}) and is used as the cross-check '
     f'for every result.')
o = fs_ov
A(OV['n_gold_overlap'] == 27 and OV['n_examples_from_gold'] == 29)
para(f'{OV["n_examples_from_gold"]} of the {OV["n_examples"]} few-shot examples were taken from {OV["n_gold_overlap"]} '
     f'gold-standard rows, most as an excerpt of the description; the others were written for the prompt. The same '
     f'examples were used for all three models, except for the two Fear examples: Mistral and DeepSeek received more '
     f'strongly worded ones than Qwen. '
     f'Scored without those {OV["n_gold_overlap"]} rows, the few-shot frame results change little (Tier 3 κ: Mistral {o["mistral_fewshot_t3"]["kappa_all"]:.3f} → '
     f'{o["mistral_fewshot_t3"]["kappa_wo"]:.3f}, Qwen {o["qwen_fewshot_t3"]["kappa_all"]:.3f} → '
     f'{o["qwen_fewshot_t3"]["kappa_wo"]:.3f}), but the few-shot sentiment results fall (Tier 1 κ: Mistral '
     f'{o["mistral_fewshot_t1"]["kappa_all"]:.3f} → {o["mistral_fewshot_t1"]["kappa_wo"]:.3f}, Qwen '
     f'{o["qwen_fewshot_t1"]["kappa_all"]:.3f} → {o["qwen_fewshot_t1"]["kappa_wo"]:.3f}), because several rare '
     f'emotional labels have their only gold rows among the examples. The choice of setting does not change.')
A(o['qwen_fewshot_t1']['kappa_wo'] > EZ['Qwen']['t1']['kappa'] and o['mistral_fewshot_t1']['kappa_wo'] < EZ['Mistral']['t1']['kappa'])
A(o['mistral_fewshot_t3']['kappa_wo'] > EZ['Mistral']['t3']['kappa'] and o['qwen_fewshot_t3']['kappa_wo'] > EZ['Qwen']['t3']['kappa'])

h2('2.3  Per-Class Reliability')
para(f'Overall scores hide large differences between labels. Figures 3 and 4 show per-class precision, recall and F1 '
     f'for Mistral in the analysis setting; Table 2 adds Qwen. Labels with 0–10 gold rows have rough scores only. The '
     f'confusion matrices are in Appendix B.')
fig(f'{BR}/fig01.png', 'Per-class precision, recall and F1, Tier 1 sentiment — Mistral 7B zero-shot. Gold rows and '
    'predictions are shown under each label.')
fig(f'{BR}/fig02.png', 'Per-class precision, recall and F1, Tier 3 frame — Mistral 7B few-shot.')
neu = P1('Neutral')
fear_n = P1('Fear')['npred']
A(R['fear_pred_gold_labels'] == {'Neutral': fear_n})
bullet(f'Precision {neu["p"]:.3f}, recall {neu["r"]:.3f}. Almost every post Mistral calls Neutral is Neutral, but it '
       f'gives an emotional label to {R["pred_emotional_gold"]:.1f}% of gold rows, against '
       f'{100 - n_neu / 4:.1f}% in the gold standard. Emotional-label shares are upper bounds.', 'Neutral: ')
bullet(f'Mistral predicted Fear for {fear_n} gold rows; the gold standard codes all {fear_n} as Neutral. Fear findings '
       f'describe what the classifier labels Fear.', 'Fear (no gold rows): ')
bullet(f'{pbm["npred"]} predictions against {pbm["support"]} gold rows (precision {pbm["p"]:.2f}); {R["mil_as_pb"]} of '
       f'the {ms["support"]} gold Military/strategic rows were labelled Political blame.', 'Political blame (over-assigned): ')
bullet(f'{ms["npred"]} predictions against {ms["support"]} gold rows (recall {ms["r"]:.2f}).',
       'Military/strategic (under-detected): ')
A(ecm['f1'] == max(P3(l)['f1'] for l in FR10))
bullet(f'Economic/sanctions among frames (F1 {ecm["f1"]:.2f}, {ecm["support"]} gold rows) and Humour among emotions '
       f'(F1 {hu["f1"]:.2f}, {hu["support"]} gold rows).', 'Most reliable: ')
qrows = []
for t, labs in (('t1', SENT9 + ['Neutral']), ('t3', FR11)):
    qrows.append(['Tier 1 (sentiment)' if t == 't1' else 'Tier 3 (frame)', '', '', '', '', ''])
    for l in labs:
        a, b = c.pcl('Mistral', t, l), c.pcl('Qwen', t, l)
        qrows.append([l, str(a['support']), str(a['npred']), f'{a["f1"]:.2f}', str(b['npred']), f'{b["f1"]:.2f}'])
table(['Label', 'Gold rows', 'Mistral pred.', 'Mistral F1', 'Qwen pred.', 'Qwen F1'], qrows,
      [2600, 1200, 1300, 1200, 1200, 1200], size=18, bold_rows=(0, len(SENT9) + 2))
caption('Table 2. Per-class F1 on the 400 gold rows — Mistral 7B (zero-shot sentiment, few-shot frame) and Qwen2.5-3B '
        'few-shot.')
para(f'Qwen\'s emotional labels are no more reliable than Mistral\'s, so agreement between the two models on an '
     f'emotional label does not show that the label is correct.')

h2('2.4  Correcting for Classification Error')
para(f'Because the gold standard is a random sample of the same videos, it shows, for each label Mistral predicts, how '
     f'often that label is right and what the true label is otherwise. The error-corrected share combines Mistral\'s '
     f'predicted mix with these rates. The bootstrap (2,000 draws) resamples both the gold rows and the videos, so the '
     f'interval covers classification error as well as sampling variation. The method assumes Mistral makes the same '
     f'kinds of errors in every group and week.')
fig(f'{BR}/fig03.png', 'Frame mix on the 400 gold rows: gold standard (with 95% bootstrap CIs) against Mistral 7B '
    'few-shot and Qwen2.5-3B few-shot.')
rows = []
for tier, labs, ov_ in (('Frame', FR11, ofr), ('Sentiment', SENT9 + ['Neutral'], osent)):
    for l in labs:
        v = ov_[l]
        corr = (f'{v["corrected"]:.1f}% [{v["ci"][0]:.1f}–{v["ci"][1]:.1f}%]' if v['gold_n'] > 0 else 'not estimable')
        rows.append([l, tier, pc(v['raw']), corr, f'{v["gold"]:.1f}% [{v["gold_exact"][0]:.1f}–{v["gold_exact"][1]:.1f}%]',
                     str(v['gold_n'])])
table(['Label', 'Tier', 'Classifier', 'Error-corrected [95% CI]', 'Gold standard [exact 95% CI]', 'Gold rows'], rows,
      [2300, 1100, 1100, 2100, 2200, 900], size=18)
caption('Table 3. Classifier shares, error-corrected shares and gold-standard shares — Mistral 7B. "Not estimable": the '
        'label has no gold rows.')
pbo, mso, hum_o = ofr['Political blame'], ofr['Military/strategic'], osent['Humour']
para(f'For frames, the correction brings the classifier close to the gold standard: Political blame falls from '
     f'{pc(pbo["raw"])} to {pc(pbo["corrected"])}, and Military/strategic rises from {pc(mso["raw"])} to '
     f'{pc(mso["corrected"])}. For sentiment, every corrected emotional share is small (Humour, the best-supported, is '
     f'{hum_o["corrected"]:.1f}% [{hum_o["ci"][0]:.1f}–{hum_o["ci"][1]:.1f}%]), and Fear and Sadness cannot be '
     f'estimated. The comparisons in Section 3 use the classifier labels with sampling intervals; Section 4.2 re-tests '
     f'the main ones with classification error included.')

h2('2.5  Non-English Posts')
para('Every post was classified from its English translation, so non-English posts depend on the translation and are '
     'more often short or hashtag-only. Table 4 compares English and non-English posts within each region.')
table(['Region', 'Unclear frame (all)', 'Unclear (English)', 'Unclear (non-English)', 'Neutral (English)',
       'Neutral (non-English)', 'Non-English n'],
      [[g, pc(ne[g]['unc_all']), pc(ne[g]['unc_en']), pc(ne[g]['unc_ne']), pc(ne[g]['neu_en']), pc(ne[g]['neu_ne']),
        str(ne[g]['n_ne'])] for g in REG8], [900, 1450, 1300, 1450, 1300, 1450, 1100], size=18)
caption('Table 4. Unclear frame and Neutral sentiment rates for English and non-English posts by region — Mistral 7B. '
        'US = United States, GB = United Kingdom, PK = Pakistan, NG = Nigeria, ID = Indonesia, AE = United Arab Emirates, '
        'FR = France, DE = Germany.')
exc = [g for g in REG8 if ne[g]['unc_ne'] <= ne[g]['unc_en']]
A(exc == ['ID'])
para(f'Non-English posts are coded Unclear far more often in every region except Indonesia, which has only '
     f'{ne["ID"]["n_en"]} English posts. The gold standard shows that most of this gap comes from the posts: it codes '
     f'{ng["gold_unc_ne"]:.1f}% of the {ng["n_ne"]} non-English rows as Unclear, against {ng["gold_unc_en"]:.1f}% of the '
     f'{ng["n_en"]} English rows, and {ng["prec_unc_ne"]:.1f}% of Mistral\'s Unclear predictions on non-English rows match '
     f'it. Where the gold standard gives a non-English post a specific frame ({ng["n_framed_ne"]} rows), Mistral matches '
     f'it in {ng["agree_framed_ne"]:.1f}% of cases, against {ng["agree_framed_en"]:.1f}% for English posts.')
para(f'Two limits apply. The coding guideline asked coders to read the English translation first, so agreement on '
     f'Unclear cannot fully separate a post with no frame from a translation that lost it. And the gold standard has few '
     f'non-English rows (17 Indonesian, 17 German, 13 French, 11 Spanish). Differences between non-English communities '
     f'are therefore reported as differences in the classifier output, without interpreting the communities themselves.')

# ── 3. Findings ──────────────────────────────────────────────────────────────
h1('3. Findings')
para(f'Labels in this section come from Mistral (zero-shot sentiment, few-shot frame), with Qwen few-shot as the '
     f'cross-check. Intervals are 95% bootstrap intervals (2,000 resamples) and cover sampling variation only.')
h2('3.1  Overall Distributions')
mn, qnn, dn = c.d('Mistral', 'sent', 'Neutral'), c.d('Qwen', 'sent', 'Neutral'), c.d('DeepSeek', 'sent', 'Neutral')
A(emo_rank[:2] == ['Fear', 'Anger'])
para(f'Neutral is the largest sentiment label for every model, from {dn:.0f}% (DeepSeek) to {mn:.0f}% (Mistral), against '
     f'{R["gold_neutral_ci"][0]:.1f}% in the gold standard. Fear and Anger are Mistral\'s most frequent emotional '
     f'labels ({ci(R["ci_sent"]["Fear"])} and {ci(R["ci_sent"]["Anger"])}), but the gold standard has no Fear rows, and '
     f'only {pr["Anger"][0]} of {pr["Anger"][1]} Anger predictions {"was" if pr["Anger"][0] == 1 else "were"} confirmed '
     f'(Section 2.3).')
top3 = sorted(FR10, key=lambda l: -c.d('Mistral', 'frame', l))[:3]
para(f'For frames, Mistral puts ' + lst(f'{l} ({ci(R["ci_frame"][l])})' for l in top3)
     + f' first. Qwen few-shot and DeepSeek both put Military/strategic first, as the gold standard does; the corrected '
       f'shares in Table 3 are the better estimate of the frame mix.')
A(max(FR10, key=lambda l: c.d('Qwen', 'frame', l)) == 'Military/strategic' and max(FR10, key=lambda l: c.d('DeepSeek', 'frame', l)) == 'Military/strategic')
fig(f'{BR}/fig06.png', 'Sentiment distribution by model — best setting per model. RoBERTa is three-class, so only its '
    'Neutral share is shown.')
fig(f'{BR}/fig07.png', 'Frame distribution by model — few-shot setting for all three LLMs.')

h2('3.2  Trends over Time')
fig(f'{BR}/fig09.png', 'Weekly sentiment shares — Mistral 7B zero-shot.')
para(f'Solidarity is highest in the first week ({pc(sol[0])}) and falls to {pc(sol[-1])} by late April; Qwen shows the '
     f'same decline ({pc(solq[0])} to {pc(solq[-1])}). Fear and Anger both peak in April 6–12 (Fear {pc(fpv)}, Anger '
     f'{pc(apv)}), a week that is mostly the April 6 window, the day before the ceasefire. Both peaks also appear in Qwen.')
A(c.peak('sent_q', 'Fear')[0] == WK[5] and c.peak('sent_q', 'Anger')[0] == WK[5])
fm_w, fq_w = W('sent_m', 'Fear'), W('sent_q', 'Fear')
fig(f'{BR}/fig17.png', 'Weekly Fear share — Mistral 7B zero-shot vs Qwen2.5-3B few-shot.')
para(f'Qwen assigns Fear about {min(b / a for a, b in zip(fm_w, fq_w)):.1f}–{max(b / a for a, b in zip(fm_w, fq_w)):.1f} '
     f'times as often as Mistral each week, but the two weekly series move together (correlation '
     f'{R["week_corr"]["Fear"]:.2f}). The models agree on the timing of Fear labels more than on their level.')
fig(f'{BR}/fig13.png', 'Weekly frame shares — Mistral 7B few-shot.')
ecw = W('frame_m', 'Economic/sanctions')
A(ecw.index(max(ecw)) == 1 and pbw.index(max(pbw)) == 5)
para(f'Political blame rises from {pc(pbw[0])} in the first week to {pc(pbw[5])} in April 6–12 (weekly correlation with '
     f'Qwen {R["week_corr"]["Political blame"]:.2f}). Economic/sanctions peaks in March 9–15 ({pc(ecw[1])}). '
     f'Humanitarian framing declines in both models (Mistral {pc(huw[0])} to {pc(huw[-1])}, Qwen {pc(huq[0])} to '
     f'{pc(huq[-1])}).')
rows = []
for lab, key, tag, l in (('Solidarity', 'Solidarity_m', 'sent_m', 'Solidarity'), ('Solidarity (Qwen)', 'Solidarity_q', 'sent_q', 'Solidarity'),
                         ('Humanitarian', 'Humanitarian_m', 'frame_m', 'Humanitarian'),
                         ('Humanitarian (Qwen)', 'Humanitarian_q', 'frame_q', 'Humanitarian'),
                         ('Political blame', 'Political_m', 'frame_m', 'Political blame'), ('Fear', 'Fear_m', 'sent_m', 'Fear'),
                         ('Anger', 'Anger_m', 'sent_m', 'Anger'), ('Historical context', 'Historical_m', 'frame_m', 'Historical context')):
    w = W(tag, l)
    t, t2 = c.tt(key), c.tt(key + '_nomonth')
    rows.append([lab, pc(w[0]), pc(w[-1]), f'{t["z"]:.2f}', pv(t['p']), f'{t2["z"]:.2f}', pv(t2['p'])])
table(['Label', 'First week', 'Final week', 'z (all)', 'p (all)', 'z (no month q.)', 'p (no month q.)'], rows,
      [2300, 1100, 1100, 1000, 1150, 1350, 1350], size=18)
caption('Table 5. Cochran–Armitage trend tests on weekly shares (negative z = decline) — Mistral 7B unless marked Qwen. '
        '"No month q.": without the month-query videos.')
ang_t, ang_t2 = c.tt('Anger_m'), c.tt('Anger_m_nomonth')
A(ang_t['p'] > 0.05 and ang_t2['p'] < 0.05)
para(f'The Solidarity and Humanitarian declines and the Political blame rise are the clearest trends; all three hold '
     f'without the month-query videos. With classification error included (Section 4.2), the Political blame rise and '
     f'the Humanitarian decline hold, while the Solidarity decline is not confirmed. Fear shows no steady trend '
     f'({fp(fear_t["p"])}); it moves with the events. Anger rises only when the month-query videos are removed '
     f'({fp(ang_t["p"])} with them, {fp(ang_t2["p"])} without), so that trend is weak.')

h2('3.3  Sentiment × Frame Co-Occurrence')
para('Row share is the percentage of a sentiment\'s videos that carry a frame. Lift compares that share with the '
     'frame\'s overall share: a lift of 3 means the frame is three times as common with that sentiment as in all videos. '
     'Fisher\'s exact test checks whether a pairing is more common than chance.')
fig(f'{BR}/fig21.png', 'Co-occurrence lift (row share ÷ overall frame share), Mistral 7B zero-shot sentiment × few-shot '
    'frame. Cells with fewer than 5 videos are blank; red = more common than overall, blue = less common.')
A(c.cotop('Anger') == 'Political blame' and c.cotop('Anger', 'q') == 'Political blame' and c.cotop('Humour') == 'Satire/meme')
A(c.cotop('Sadness') == 'Humanitarian' and c.cotop('Solidarity') == 'Humanitarian' and c.cotop('Fear') == 'Military/strategic')
A(c.cotop('Fear', 'q') == 'Military/strategic')
ap_ci = R['cooc_ci']['Anger']['Political blame']
bullet(f'{pc(hs["pct"])} of Humour videos (lift {hs["lift"]:.0f}, Fisher {fp(hs["p"])}). This is the best-supported '
       f'pairing: {pr["Humour"][0]} of {pr["Humour"][1]} Humour predictions were confirmed on the gold standard.',
       'Humour → Satire/meme: ')
bullet(f'{pc(ap["pct"])} [{ap_ci[1]:.1f}–{ap_ci[2]:.1f}%] of Anger videos (lift {ap["lift"]:.1f}); the top frame for '
       f'Anger in both models and all three settings.', 'Anger → Political blame: ')
sa, so_ = c.co('Sadness', 'Humanitarian'), c.co('Solidarity', 'Humanitarian')
bullet(f'Sadness {pc(sa["pct"])} (lift {sa["lift"]:.0f}) and Solidarity {pc(so_["pct"])} (lift {so_["lift"]:.0f}).',
       'Sadness and Solidarity → Humanitarian: ')
fm_ = c.co('Fear', 'Military/strategic')
bullet(f'{pc(fm_["pct"])} of Fear videos (lift {fm_["lift"]:.1f}), in both models.', 'Fear → Military/strategic: ')
para(f'Both labels in each pairing come from the classifier, so these patterns show how Mistral links its own labels. '
     f'Only {pr["Anger"][0]} of {pr["Anger"][1]} Anger, {pr["Solidarity"][0]} of {pr["Solidarity"][1]} Solidarity and '
     f'{pr["Fear"][0]} of {pr["Fear"][1]} Fear predictions were confirmed on the gold standard, so pairings with these '
     f'labels are classifier patterns, not evidence of how creators combine emotion and framing.')

h2('3.4  Language and Region')
lc, rc = R['lang_chi'], R['reg_chi']
para(f'Across the six largest languages the frame mix differs (χ² {fp(lc["frame_label"]["p"])}, Cramér\'s V '
     f'{lc["frame_label"]["V"]:.2f}), mostly through the Unclear share. Across the eight largest regions the differences '
     f'are small (sentiment V {rc["sentiment_label"]["V"]:.2f}, frame V {rc["frame_label"]["V"]:.2f}).')
fig(f'{BR}/fig24.png', 'Frame by language — Mistral 7B few-shot.')
fig(f'{BR}/fig28.png', 'Frame by region — Mistral 7B few-shot.')
fig(f'{BR}/fig25.png', 'Religious framing by language with 95% bootstrap confidence intervals — Mistral 7B few-shot.')
fig(f'{BR}/fig29.png', 'Religious framing by region with 95% bootstrap confidence intervals — Mistral 7B few-shot.')
lci, rci = R['lang_ci'], R['reg_ci']
A(c.top('lang', 'frame', 'Religious framing') == 'Indonesian' and id_rel['p'] < 0.001)
A(c.sh('lang', 'frame', 'Indonesian', 'Religious framing', 'q') > c.sh('lang', 'frame', 'English', 'Religious framing', 'q'))
A(c.rank('reg', 'frame', 'Religious framing', 'NG') == 2 and ng_rel['p'] < 0.01)
A(R['nonenglish']['NG']['n_en'] / R['reg_n']['NG'] > 0.85)
para(f'Religious framing is {ci(lci["Religious framing"]["Indonesian"])} of Indonesian-language posts, '
     f'{times(id_rel["pa"], id_rel["pb"])} the English rate ({ci(lci["Religious framing"]["English"])}; Fisher '
     f'{fp(id_rel["p"])}); Qwen shows the same direction. The difference holds with classification error included '
     f'(corrected {pc(rel_c["corr_a"])} vs {pc(rel_c["corr_b"])}, {fp(rel_c["p"])}). Nigerian posts are second among '
     f'regions ({ci(rci["Religious framing"]["NG"])} against {pc(ng_rel["pb"])} for the US; corrected {fp(ngr_c["p"])}). '
     f'{R["nonenglish"]["NG"]["n_en"]} of the {R["reg_n"]["NG"]} Nigerian posts are in English, so this is not a '
     f'translation effect. These results show how often posts use religious framing, not why.',
     'Religious framing: ')
para(f'In the classifier output, British posts have the highest Solidarity share of the eight regions '
     f'({ci(rci["Solidarity"]["GB"])}, against {pc(gb_sol["pb"])} for the US), and Indonesian and French posts have more '
     f'Historical context than English posts. None of these differences is confirmed once classification error is '
     f'included (Section 4.2): Solidarity and Historical context each have only 3 gold rows.',
     'Not confirmed: ')
para(f'German and Catalan posts are {pc(c.sh("lang", "frame", "German", "Unclear"))} and '
     f'{pc(c.sh("lang", "frame", "Catalan", "Unclear"))} Unclear, and Pakistani posts {pc(c.sh("reg", "frame", "PK", "Unclear"))}. '
     f'Section 2.5 shows that this mostly reflects short or hashtag-only posts, so these communities are not interpreted. '
     f'French vs English Fear and Anger, US vs German Fear and UAE vs US Anger do not differ significantly '
     f'(Appendix E).', 'Unclear-heavy communities: ')
A(all(c.pt(k)['p'] > 0.05 for k in ('French|English|Fear', 'French|English|Anger', 'US|DE|Fear', 'AE|US|Anger')))

h2('3.5  Engagement')
kw = R['kw']
para(f'Medians are used for likes and views because a few viral videos dominate the means. Each label is compared with '
     f'all other labelled videos (Mann–Whitney U). Engagement differs across sentiment labels and across frames '
     f'(Kruskal–Wallis, all {fp(max(v["p"] for v in kw.values()))}). Groups are defined by the classifier\'s labels.')
fig(f'{BR}/fig33.png', 'Median likes by frame with 95% bootstrap CIs — Mistral 7B few-shot.')
fig(f'{BR}/fig34.png', 'Median views by frame with 95% bootstrap CIs — Mistral 7B few-shot.')
lf = sorted(FR11, key=lambda l: -c.ef(l)['med_likes'])
vf = sorted(FR11, key=lambda l: -c.ef(l)['med_views'])
A(lf[-1] == 'Economic/sanctions' and e_['mw_likes_p'] < 0.001 and vf[0] == 'Military/strategic' and mi['mw_views_p'] < 0.001)
A(min(FR11, key=lambda l: RES['best']['eng_frame_q'][l]['med_likes']) == 'Economic/sanctions')
bullet(f'Economic/sanctions posts receive the fewest likes (median {num(e_["med_likes"])}, {fp(e_["mw_likes_p"])}) in '
       f'both Mistral and Qwen, and below-average views (median {num(e_["med_views"])}). With {k_ec} of {n_ec} '
       f'predictions confirmed on the gold standard, this is the most reliable engagement result.')
bullet(f'Military/strategic posts receive the most views (median {num(mi["med_views"])}, {fp(mi["mw_views_p"])}). The '
       f'frame is under-detected (recall {ms["r"]:.2f}), so the posts it captures may not be typical of the frame.')
p_ = c.ef('Personal connection')
A(lf[0] == 'Personal connection')
bullet(f'Personal connection has the highest median likes ({num(p_["med_likes"])}), but it is over-predicted '
       f'({P3("Personal connection")["npred"]} predictions against {P3("Personal connection")["support"]} gold rows), so '
       f'this is tentative.')
fig(f'{BR}/fig31.png', 'Median likes by sentiment with 95% bootstrap CIs — Mistral 7B zero-shot.')
ml = sorted(SENT9 + ['Neutral'], key=lambda l: -c.es(l)['med_likes'])
A(ml[0] == 'Solidarity' and s_['mw_likes_p'] < 0.001)
fe = c.es('Fear')
bullet(f'Videos labelled Solidarity receive the most likes (median {num(s_["med_likes"])} [{nr(s_["likes_ci"][1])}–'
       f'{nr(s_["likes_ci"][2])}], {fp(s_["mw_likes_p"])}), and videos labelled Fear the most views (median '
       f'{num(fe["med_views"])}, {fp(fe["mw_views_p"])}). Only {pr["Solidarity"][0]} of {pr["Solidarity"][1]} Solidarity '
       f'and {pr["Fear"][0]} of {pr["Fear"][1]} Fear predictions were confirmed, so these describe the classifier\'s '
       f'labels.')

# ── 4. Robustness ────────────────────────────────────────────────────────────
h1('4. Robustness Checks')
h2('4.1  Across Prompt Settings')
para('The analysis was repeated in two strict settings, where both tiers come from one prompt setting (all zero-shot, '
     'all few-shot). Table 6 shows which findings survive a change of setting.')
table(['Finding', 'Best setting', 'Strict zero-shot', 'Strict few-shot'], robustness_rows(), [3300, 2000, 2000, 2000],
      size=18)
caption('Table 6. Key findings under the three prompt settings (Mistral 7B unless stated).')
C3 = {s: Ctx(s) for s in ('best', 'zs', 'fs')}
A(all(C3[s].cotop('Anger', 'q') == 'Political blame' for s in C3))
A(C3['zs'].d('Mistral', 'frame', 'Political blame') > C3['best'].d('Mistral', 'frame', 'Political blame') + 5)
A(C3['fs'].d('Mistral', 'sent', 'Neutral') < C3['best'].d('Mistral', 'sent', 'Neutral') - 30)
A(all(C3[s].cotop('Anger') == 'Political blame' for s in C3) and all(C3[s].cotop('Humour') == 'Satire/meme' for s in C3))
A(all(C3[s].tt('Humanitarian_m')['p'] < 0.001 and C3[s].pt('Indonesian|English|Religious framing')['p'] < 0.001 for s in C3))
A(all(min(FR11, key=lambda l: C3[s].ef(l)['med_likes']) == 'Economic/sanctions' for s in C3))
para('Anger → Political blame, Humour → Satire/meme, the Humanitarian decline, Indonesian Religious framing and the low '
     'engagement of Economic/sanctions posts hold in all three settings. The frame shares depend on the setting: strict '
     'zero-shot over-assigns Political blame even more. Strict few-shot changes the sentiment results most, because '
     'few-shot Mistral assigns far fewer Neutral labels.')
h2('4.2  With Classification Error')
para('Each main comparison was re-tested with the error-corrected shares from Section 2.4. A finding "holds" when the '
     '95% interval of the corrected difference excludes zero. "Not confirmed" does not mean the difference is absent: '
     'for labels with only a few gold rows, the gold standard is too small to confirm or reject it.')
rows = []
for name, v in E['compare'].items():
    if name.startswith('Fear'):
        continue
    rows.append([name, f'{v["raw_a"]:.1f}% vs {v["raw_b"]:.1f}%', f'{v["corr_a"]:.1f}% vs {v["corr_b"]:.1f}%',
                 f'{v["diff"]:+.1f} [{v["diff_ci"][0]:+.1f}, {v["diff_ci"][1]:+.1f}]', pv(v['p']),
                 'Holds' if v['p'] < 0.05 else 'Not confirmed'])
table(['Comparison', 'Classifier', 'Error-corrected', 'Difference [95% CI]', 'p', 'Result'], rows,
      [3000, 1400, 1400, 1700, 800, 1400], size=17)
caption('Table 7. Main comparisons with classification error included — Mistral 7B zero-shot sentiment, few-shot frames. '
        'Week comparisons: first two weeks (March 2–15) against last two (April 20 – May 3).')
held = [n for n, v in E['compare'].items() if v['p'] < 0.05]
A(set(held) == {'Religious framing: Indonesian vs English', 'Religious framing: ID vs US', 'Religious framing: NG vs US',
                'Humanitarian: first two weeks vs last two', 'Political blame: first two weeks vs last two'})
para(f'Religious framing for Indonesian and Nigerian posts, the Political blame rise and the Humanitarian decline hold. '
     f'The corrected Religious framing shares are about half the classifier\'s ({pr["Religious framing"][0]} of '
     f'{pr["Religious framing"][1]} predictions confirmed), but the gap between groups remains. Fear comparisons cannot '
     f'be tested, because the gold standard contains no Fear.')

# ── 5. Key findings ──────────────────────────────────────────────────────────
h1('5. Key Findings')
bullet(f'Open-source LLMs classify narrative frames moderately well (Tier 3 κ up to {EF["Qwen"]["t3"]["kappa"]:.2f} with '
       f'full coverage) and sentiment poorly (Tier 1 κ at most '
       f'{max(v["t1"]["kappa"] for v in list(EZ.values()) + list(EF.values())):.2f}). Few-shot prompting improves frames '
       f'for every model; for sentiment it helps only Qwen.')
bullet(f'Emotion is rare in these descriptions: the gold standard is {R["gold_neutral_ci"][0]:.1f}% Neutral and contains '
       f'no Fear. Sentiment patterns in this report are patterns in the classifier\'s labels.')
bullet(f'Corrected for classification error, Military/strategic is the most common frame '
       f'({ofr["Military/strategic"]["corrected"]:.1f}%), followed by Economic/sanctions and Political blame '
       f'({ofr["Political blame"]["corrected"]:.1f}%). The raw classifier output over-assigns Political blame.')
bullet('Political blame rises and Humanitarian framing falls between early March and late April, and both trends hold '
       'with classification error included. Fear and Anger labels peak in the April 6 window, just before the '
       'ceasefire, in both models.')
bullet(f'Religious framing is more common in Indonesian-language posts than in English posts and in Nigerian posts than '
       f'in US posts, also after correcting for classification error.')
bullet(f'Humour → Satire/meme is the best-supported co-occurrence (lift {hs["lift"]:.0f}). Anger → Political blame holds '
       f'in every setting but rests on a weakly validated label.')
bullet(f'Economic/sanctions posts, the most reliably classified frame, receive the fewest likes; Military/strategic posts '
       f'the most views.')
bullet('High Unclear rates in non-English content mostly reflect short or hashtag-only posts, not classifier failure.')

# ── 6. Limitations ───────────────────────────────────────────────────────────
h1('6. Limitations')
bullet(f'The gold standard has {n_neu} Neutral rows out of 400. Emotional labels have 0–10 gold rows each, and Fear, '
       f'Sadness and Misinformation have none, so their reliability cannot be measured.')
bullet('Classification error is carried into the overall shares and the main comparisons only (Sections 2.4 and 4.2); '
       'other percentages carry sampling uncertainty only. The correction assumes the same error pattern in every group '
       f'and week, which cannot be checked for individual non-English communities ({ng["n_ne"]} non-English gold rows).')
bullet(f'{OV["n_examples_from_gold"]} of the {OV["n_examples"]} few-shot examples were taken from gold-standard rows, '
       f'which makes the few-shot scores optimistic, above all for sentiment (Section 2.2). Qwen received different Fear '
       f'examples from Mistral and DeepSeek, so few-shot Fear rates are not strictly comparable across models.')
bullet('The sample is nine two-day windows plus month-end bulk queries, not a continuous record. Weekly results describe '
       'those windows, and events between windows are not observed.')
bullet('Posts were classified from English translations, and the gold standard was also coded from the translations '
       'first. German, Catalan and Pakistani results rest largely on short or hashtag-only posts.')
bullet(f'DeepSeek-R1 classification is partial ({min(dsn):,}–{max(dsn):,} of 5,848 videos), so it is used only in the '
       f'benchmark and the overall distributions.')
bullet('The analysis covers video descriptions only, not video content, audio or images.')
bullet('The cleaning step kept only ASCII characters, before language detection and translation. Posts written in '
       'non-Latin scripts (Persian, Arabic, Russian, Hebrew, Chinese and others) were rare in the collected data (52 raw '
       'rows) and lost their text, so the analysed data contain no Persian or Arabic posts. The same step removed accented '
       'letters (é, ü, ñ). In the third of the videos that could be matched back to the raw files, this affected about 2% '
       'of posts: under 1% of English and Indonesian posts, about 5% of German and 8% of French posts, but most posts '
       'labelled Catalan, so the Catalan language label itself is unreliable.')
bullet('TikTok\'s recommendation system and the API\'s search results shape the dataset, so its geographic and language '
       'mix reflects curation as much as public interest.')
import pandas as pd
_lang = pd.read_csv('data/clean/iran_crisis_videos_classified_mistral.csv', usecols=['language'], low_memory=False)['language']
A(not _lang.isin(['fa', 'ar', 'ru', 'he', 'zh-cn', 'zh-tw', 'ja', 'ko']).any())

# ── Appendices ───────────────────────────────────────────────────────────────
DEFS_S = [('Fear', 'Worry, anxiety, dread about escalation or safety'),
          ('Anger', 'Frustration, rage, blame directed at a person or group'),
          ('Sadness', 'Grief, mourning, empathy for victims'),
          ('Hope', 'Optimism, support, belief in peaceful resolution'),
          ('Solidarity', 'Active support for a people or cause'),
          ('Frustration', 'Helplessness, fed-up feeling'),
          ('Humour', 'Jokes, memes, sarcasm, ironic detachment'),
          ('Disgust', 'Moral revulsion at actions or actors'),
          ('Curiosity', 'Seeking information, expressing confusion'),
          ('Neutral', 'Factual statement, no detectable emotion')]
DEFS_F = [('Political blame', 'Assigning responsibility to a government or person'),
          ('Military/strategic', 'Discussing weapons, tactics, or power balance'),
          ('Economic/sanctions', 'Focus on oil prices, trade, or economic impact'),
          ('Humanitarian', 'Focus on civilian suffering and human cost'),
          ('Religious framing', 'Invoking religion or divine judgment'),
          ('Satire/meme', 'Humorous or ironic detachment'),
          ('Historical context', 'Connecting current events to past history'),
          ('Media criticism', 'Criticizing how news covers the crisis'),
          ('Personal connection', 'Speaker has family or friends in the affected area'),
          ('Misinformation', 'A specific factual claim about the crisis that is false, conspiratorial, or unverifiable'),
          ('Unclear', 'Off-topic, too short, garbled, or does not relate to any frame above')]
A(sum(R['gold_counts']['t1'].values()) == 400 and 'Unclear' not in R['gold_counts']['t1'])
h1('Appendix A. Label Definitions', page_break=True)
table(['Tier 1 label', 'Definition'], [[a, b] for a, b in DEFS_S], [2400, 7000], size=19)
caption('Table A1. Tier 1 sentiment labels, as worded in the few-shot prompts. In the model output, Unclear '
        'marks posts that received no valid label.')
table(['Tier 3 label', 'Definition'], [[a, b] for a, b in DEFS_F], [2400, 7000], size=19)
caption('Table A2. Tier 3 frame labels, as worded in the few-shot prompts.')

h1('Appendix B. Confusion Matrices')
fig(f'{FX}/cm_t1.png', 'Confusion matrix, Tier 1 sentiment — Mistral 7B zero-shot, 400 gold rows. Rows: gold-standard '
    'label; columns: predicted label. Labels with no gold rows and no predictions are omitted.', label='Figure B1')
fig(f'{FX}/cm_t3.png', 'Confusion matrix, Tier 3 frame — Mistral 7B few-shot, 400 gold rows.', label='Figure B2')

h1('Appendix C. Full Evaluation Scores')
rows = []
for m, nm in (('Mistral', 'Mistral 7B'), ('Qwen', 'Qwen2.5-3B'), ('DeepSeek', 'DeepSeek-R1'), ('RoBERTa', 'RoBERTa')):
    for E_, sname in ((EZ, 'Zero-shot'), (EF, 'Few-shot')):
        if m not in E_:
            continue
        for t in ('t1', 't3'):
            if t not in E_[m]:
                continue
            v = E_[m][t]
            rows.append([nm, sname, 'Tier 1' if t == 't1' else 'Tier 3', f'{v["kappa"]:.3f}', f'{v["wf1"]:.3f}',
                         f'{v["mf1"]:.3f}', f'{v["agree"]:.1f}%'])
table(['Model', 'Setting', 'Tier', 'κ', 'Weighted F1', 'Macro F1', 'Agreement'], rows,
      [2000, 1300, 1000, 1000, 1400, 1300, 1300], size=18)
caption('Table C1. All evaluation scores against the gold standard. DeepSeek-R1 is scored on the gold rows it covers '
        f'({EZ["DeepSeek"]["n_t1"]} zero-shot, {EF["DeepSeek"]["n_t1"]} few-shot); RoBERTa on three classes.')

h1('Appendix D. Label Shares and Weekly Shares')
rows = [[l, ci(R['ci_sent'][l]), pc(c.d('Qwen', 'sent', l)), pc(c.d('DeepSeek', 'sent', l))] for l in SENT9 + ['Neutral', 'Unclear']]
table(['Sentiment', 'Mistral zero-shot [95% CI]', 'Qwen few-shot', 'DeepSeek (partial)'], rows, [2200, 2900, 1900, 2000], size=18)
caption('Table D1. Sentiment shares (% of videos).')
rows = [[l, ci(R['ci_frame'][l]), pc(c.d('Qwen', 'frame', l)), pc(c.d('DeepSeek', 'frame', l)), ci(gfc[l])] for l in FR11]
table(['Frame', 'Mistral few-shot [95% CI]', 'Qwen few-shot', 'DeepSeek (partial)', 'Gold standard [95% CI]'], rows,
      [2300, 2300, 1300, 1400, 2300], size=18)
caption('Table D2. Frame shares (% of videos; gold standard = % of the 400 gold rows).')
for tag, labs, nm in (('sent_m', ['Fear', 'Anger', 'Solidarity', 'Humour'], 'sentiment'),
                      ('frame_m', ['Political blame', 'Military/strategic', 'Economic/sanctions', 'Humanitarian'], 'frame')):
    rows = []
    for i, w in enumerate(c.wk):
        rows.append([WK[i], f'{wt[w]:,}', f'{wtn[w]:,}']
                    + [f'{R["weekly"][tag][l][w]:.1f} ({R["weekly"][tag + "_nomonth"][l][w]:.1f})' for l in labs])
    table(['Week', 'Videos', 'No month q.'] + labs, rows, [1900, 900, 1100] + [1350] * 4, size=17)
    caption(f'Table D{3 if nm == "sentiment" else 4}. Weekly {nm} shares (%) — Mistral 7B. In brackets: without '
            f'month-query videos.')

h1('Appendix E. Pairwise Comparisons')
rows = []
for k, v in R['pair_tests'].items():
    a, b, lab = k.split('|')
    rows.append([lab, f'{a} vs {b}', f'{v["pa"]:.1f}% ({v["a"]}/{v["na"]:,})', f'{v["pb"]:.1f}% ({v["b"]}/{v["nb"]:,})', pv(v['p'])])
table(['Label', 'Comparison', 'Group A', 'Group B', 'Fisher p'], rows, [2100, 2200, 1800, 1900, 1200], size=17)
caption('Table E1. Fisher\'s exact tests for the community comparisons in Section 3.4 (classifier labels).')

h1('Appendix F. Engagement')
SE = SENT9 + ['Neutral']
for key, labs, nm in (('eng_sent_m', SE, 'sentiment'), ('eng_frame_m', FR11, 'frame')):
    rows = []
    for l in labs:
        v = R[key][l]
        rows.append([l, f'{v["n"]:,}', f'{num(v["med_likes"])} [{nr(v["likes_ci"][1])}–{nr(v["likes_ci"][2])}]',
                     pv(v['mw_likes_p']), f'{num(v["med_views"])} [{nr(v["views_ci"][1])}–{nr(v["views_ci"][2])}]',
                     pv(v['mw_views_p']), f'{v["mean_shares"]:.1f}', f'{v["mean_comments"]:.1f}'])
    table([nm.capitalize(), 'n', 'Median likes [CI]', 'p', 'Median views [CI]', 'p', 'Mean shares', 'Mean comm.'],
          rows, [1900, 750, 1550, 850, 1700, 850, 950, 950], size=16)
    caption(f'Table F{1 if nm == "sentiment" else 2}. Engagement by {nm} — Mistral 7B. p = Mann–Whitney U against all '
            f'other labelled videos.')

h1('Appendix G. Data Notes')
bullet(f'{R["n_no_date"]} videos have no posting date and are left out of the time analysis. 4 videos have an unreadable '
       f'collection-query label and are treated as window-query videos; this does not affect any result.')
bullet('Weeks run Monday to Sunday. The week of February 23 – March 1 holds only 8 videos and is left out; the nine weeks '
       'from March 2 to May 3 each have at least 30 videos. The final week only has data for April 27–30.')
bullet('Gold rows were matched to classified videos on the first 18 digits of the video id. "Humor" in the model output '
       'was renamed "Humour".')
bullet(f'The {R["short"]["n"]} short descriptions (under 10 characters or 4 words) were labelled Unclear without a model '
       f'call in the few-shot runs only; in the best setting they affect the frame labels only.')
bullet('Raw rows that the CSV reader could not parse were skipped when the data were loaded.')
bullet('Region codes with fewer than 50 videos are not compared. Many tests are run, so p-values between 0.01 and 0.05 '
       'should be read with caution.')

doc.save(OUT)
print('saved', OUT, 'figures', len(FIGS))
