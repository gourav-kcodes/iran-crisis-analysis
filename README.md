# Iran Crisis on TikTok: Sentiment and Narrative Framing

The project classifies
5,848 TikTok video descriptions about the 2026 Iran crisis (1 March - 30 April 2026) by sentiment and by narrative
frame with open-source language models, evaluates the models against a 400-row human-coded gold standard, and
analyses the results over time, by language and region, and by engagement.

- **Tier 1 (sentiment):** Fear, Anger, Sadness, Hope, Solidarity, Frustration, Humour, Disgust, Curiosity, Neutral
- **Tier 3 (narrative frame):** Political blame, Military/strategic, Economic/sanctions, Humanitarian, Religious framing,
  Satire/meme, Historical context, Media criticism, Personal connection, Misinformation, Unclear
- **Models:** Mistral-7B-Instruct-v0.3, Qwen2.5-3B-Instruct, DeepSeek-R1-0528-Qwen3-8B (zero-shot and few-shot),
  Twitter-RoBERTa (3-class sentiment baseline)

## No data in this repository

The repository contains code only. TikTok data, the annotated gold standard, model outputs, figures and reports are not
included. The few-shot examples are also left out because most of them are real TikTok
descriptions; `04_classification/few_shot_examples_TEMPLATE.py` shows the structure.

The scripts expect this layout, relative to the repository root:

```
data/raw/      iran_crisis_videos.csv, iran_crisis_videos2.csv          (TikTok Research API exports, from Dr. Haensch)
data/clean/    iran_crisis_videos_all_clean.csv                          (output of 02_preprocessing)
               iran_crisis_videos_classified_<model>[_fewshot].csv      (outputs of 04_classification)
               gold_standard_dataset.csv                                 (400 coded rows)
               annotation_*.csv, kappa_results_full.txt                  (coder files, 03_annotation)
data/analysis/ report_best|zs|fs/, final/                                (written by 06_analysis)
```

## Pipeline

Run every script from the repository root.

| Step | Script | What it does | Main output |
|---|---|---|---|
| 1 | `01_collection/tiktok_collection_example.py` | Example of the TikTok Research API calls. The project data were collected by Dr. Haensch (see note below) | `iran_crisis_videos.csv` |
| 2 | `02_preprocessing/preprocessing.py` | Cleans text, removes duplicates, short and spam posts, detects language, translates to English (Helsinki-NLP opus-mt-mul-en). Run once per raw file (set `INPUT_FILE`/`OUTPUT_FILE`) | `*_clean.csv` |
| 2 | `02_preprocessing/merge_videos.py` | Merges the two cleaned video files and removes duplicates | `iran_crisis_videos_all_clean.csv` (5,848 videos) |
| 3 | `03_annotation/sample_for_annotation.py` | Draws the 400-row random sample for coding (seed 42) | `annotation_sample_400.csv` |
| 3 | `03_annotation/cohen_kappa_calculation.py` | Agreement between the two coders (Cohen's κ, per label) | `kappa_results_full.txt` |
| 4 | `04_classification/roberta_classification.py` | Twitter-RoBERTa 3-class sentiment | `..._classified_roberta.csv` |
| 4 | `04_classification/llm_zeroshot.py` | Zero-shot run for `mistral` or `qwen`, both tiers | `..._classified_<model>.csv` |
| 4 | `04_classification/llm_zeroshot_deepseek.py` | DeepSeek-R1 zero-shot, both tiers | `..._classified_deepseek.csv` |
| 4 | `04_classification/llm_fewshot.py` | Few-shot run for `mistral`, `qwen` or `deepseek` (2 examples per label) | `..._classified_<model>_fewshot.csv` |
| 5 | `05_evaluation/evaluation.py` | Zero-shot models against the gold standard | `evaluation_results.txt` |
| 5 | `05_evaluation/prompt_sensitivity.py` | Qwen zero-shot on 200 videos with the original and a reworded prompt | `prompt_sensitivity_results.csv` |
| 5 | `05_evaluation/fewshot_overlap.py` | Few-shot examples taken from gold rows, and scores without those rows | `final/fewshot_overlap.json` |
| 6 | `06_analysis/analysis_core.py` | All statistics for one setting: reliability, shares and CIs, weekly trends, co-occurrence, language, region, engagement, tests | `report_<s>/results.json` |
| 6 | `06_analysis/classification_error.py` | Error-corrected shares and comparisons using the gold standard as a calibration sample | `report_<s>/classification_error.json` |
| 6 | `06_analysis/report_figs.py` | The 36 figures of the analysis reports | `report_<s>/figNN.png` |
| 6 | `06_analysis/final_figs.py` | Benchmark chart and confusion matrices for the final report | `final/*.png` |
| 7 | `07_reports/build_reports.py` | Builds one analysis report (best, strict zero-shot or strict few-shot) as .docx | report .docx |
| 7 | `07_reports/final_report.py` | Builds the final combined report as .docx | `Final_Report.docx` |
| 7 | `07_reports/verify3.py`, `07_reports/verify_final.py` | Recompute key numbers from the CSVs independently and check them against the report text | check log |

`<s>` is the prompt setting: `best` (Mistral zero-shot sentiment + few-shot frame, the main analysis), `zs` (all
zero-shot) or `fs` (all few-shot). Every number in the report text is read from the JSON files, and the text scripts
stop with an error if a ranking or significance claim no longer holds.

### Reproducing the reports

```bash
pip install -r requirements.txt
for s in best zs fs; do
  python 06_analysis/analysis_core.py $s
  python 06_analysis/classification_error.py $s
  python 06_analysis/report_figs.py $s
done
python 06_analysis/final_figs.py
python 05_evaluation/fewshot_overlap.py          # needs 04_classification/few_shot_examples.py
python 07_reports/final_report.py <template.docx> Final_Report.docx
python 07_reports/verify_final.py Final_Report.docx
```

`<template.docx>` is any earlier report .docx; only its page setup and styles are used.

## Known issues

- **Cleaning removes non-ASCII characters.** `preprocessing.py` reads the raw CSV with `encoding="unicode_escape"` and
  keeps only ASCII characters, before language detection and translation. Posts in non-Latin scripts (52 raw rows)
  lost their text and were dropped, and accented letters (é, ü, ñ) were removed, which mostly affects German, French
  and Catalan posts. The script is kept as it was run so that it reproduces the dataset used in the reports.
- **Unparsable raw rows are skipped** when the raw CSV is loaded (`on_bad_lines="skip"`).
- **Data collection.** The TikTok data were collected by Dr. Anna-Carolina Haensch with the TikTok Research API:
  18 hashtags, nine two-day query windows (1–2 March to 26–27 April 2026) and two whole-month queries.
  `tiktok_collection_example.py` was written at the start of the project to test the API calls and is not the query
  set used for the final data.
- **DeepSeek-R1 is partial:** 2,700 videos zero-shot and 3,000 few-shot.
- **Zero-shot script for Mistral and Qwen.** The original script file was not kept. `llm_zeroshot.py` is rebuilt
  from the project notebooks, using the prompt that the prompt-sensitivity notebook marks as the zero-shot prompt.
