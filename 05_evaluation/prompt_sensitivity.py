# Prompt-sensitivity test (Qwen2.5-3B, zero-shot): the same 200 videos are classified with the original prompt (A)
# and a reworded prompt (B), and agreement between the two runs is reported. Exported from the Kaggle notebook used in
# the project; Kaggle paths were replaced by command-line arguments.
#
# Usage (project root):
#   python 05_evaluation/prompt_sensitivity.py data/clean/iran_crisis_videos_all_clean.csv data/clean/prompt_sensitivity_results.csv
import os, sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
INPUT_FILE, OUTPUT_FILE = sys.argv[1], sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
model.eval()

# ── PROMPT A — original (the zero-shot prompt) ────────────────────────────────
SENTIMENT_PROMPT_A = """You are a social science researcher analyzing public reactions to the Iran crisis on TikTok.
 
Classify the following text into EXACTLY ONE of these categories:
- Fear: worry, anxiety, dread about escalation or safety
- Anger: frustration, rage, blame directed at a person or group
- Sadness: grief, mourning, empathy for victims
- Hope: optimism, support, belief in peaceful resolution
- Solidarity: active support for a people or cause
- Frustration: helplessness, fed-up feeling
- Humor: jokes, memes, sarcasm, ironic detachment
- Disgust: moral revulsion at actions or actors
- Curiosity: seeking information, expressing confusion
- Neutral: factual statement, no detectable emotion
 
Text: "[TEXT HERE]"
 
Respond with ONLY the label. Nothing else."""

FRAME_PROMPT_A = """You are a social science researcher analyzing public reactions to the Iran crisis on TikTok.
 
Classify the following text into EXACTLY ONE narrative frame:
- Humanitarian: focus on civilian suffering and human cost
- Political blame: assigning responsibility to a government or person
- Military/strategic: discussing weapons, tactics, or power balance
- Personal connection: speaker has family or friends in the affected area
- Economic/sanctions: focus on oil prices, trade, or economic impact
- Historical context: connecting current events to past history
- Religious framing: invoking religion or divine judgment
- Media criticism: criticizing how news covers the crisis
- Misinformation: text makes a specific factual claim about the crisis that is false, conspiratorial, or unverifiable
- Satire/meme: humorous or ironic detachment
- Unclear: text is off-topic, too short, garbled, or does not relate to any frame above
 
Text: "[TEXT HERE]"
 
Respond with ONLY the label. Nothing else."""

# ── PROMPT B — alternative wording ────────────────────────────────────────────
SENTIMENT_PROMPT_B = """You are coding TikTok video descriptions about the Iran crisis for a social science study.
 
Choose the single most appropriate emotion label for the AUTHOR of the following text:
- Fear: author expresses personal worry or dread about what might happen
- Anger: author expresses rage or blame toward a specific target
- Sadness: author expresses mourning or grief for those suffering
- Hope: author expresses optimism or wish for a peaceful outcome
- Solidarity: author expresses active support for a group or cause
- Frustration: author expresses helplessness or exhaustion with no specific target
- Humor: author uses jokes, irony, or memes to comment on the crisis
- Disgust: author expresses moral revulsion at actions or actors
- Curiosity: author asks questions or seeks information
- Neutral: author makes a factual statement with no personal emotional expression
 
Text: "[TEXT HERE]"
 
Reply with ONLY the label name. Do not explain."""

FRAME_PROMPT_B = """You are coding TikTok video descriptions about the Iran crisis for a social science study.
 
Choose the single most appropriate narrative frame for the following text:
- Humanitarian: the main focus is on civilian casualties, suffering, or displacement
- Political blame: the text assigns responsibility to a specific government, leader, or political actor
- Military/strategic: the text discusses weapons, military operations, tactics, or geopolitical power
- Personal connection: the author mentions their own family or personal ties to the affected region
- Economic/sanctions: the text focuses on oil prices, trade, sanctions, or economic consequences
- Historical context: the text draws parallels to past events or historical patterns
- Religious framing: religion is the central lens through which the crisis is interpreted
- Media criticism: the text criticizes how mainstream or social media covers the crisis
- Misinformation: the text makes a specific false or conspiratorial factual claim
- Satire/meme: the text uses humor, irony, or pop-culture references about the crisis
- Unclear: the text is too vague, short, or off-topic to classify into any frame above
 
Text: "[TEXT HERE]"
 
Reply with ONLY the label name. Do not explain."""

VALID_SENTIMENT = [
    "Fear", "Anger", "Sadness", "Hope", "Solidarity",
    "Frustration", "Humor", "Disgust", "Curiosity", "Neutral"
]
 
VALID_FRAMES = [
    "Humanitarian", "Political blame", "Military/strategic",
    "Personal connection", "Economic/sanctions", "Historical context",
    "Religious framing", "Media criticism", "Misinformation",
    "Satire/meme", "Unclear"
]


def classify_text(text, prompt_template, valid_labels):
    if not isinstance(text, str) or len(text.strip()) < 3:
        return "Unclear"
    try:
        prompt = prompt_template.replace("[TEXT HERE]", text[:1000])
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
 
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
 
        new_tokens = output[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
 
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        label = lines[0] if lines else "Unclear"
 
        for valid in valid_labels:
            if valid.lower() in label.lower():
                return valid
        return "Unclear"
 
    except Exception as e:
        print(f"  Error: {e}")
        return "Error"


df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
sample = df.sample(n=200, random_state=123).reset_index(drop=True)
print(f"Sample selected: {len(sample)} rows")

print("\n--- Running Prompt A (original) ---")
sent_A = []
frame_A = []
 
for i, row in sample.iterrows():
    text = str(row.get("video_description_english", ""))
    s = classify_text(text, SENTIMENT_PROMPT_A, VALID_SENTIMENT)
    f = classify_text(text, FRAME_PROMPT_A, VALID_FRAMES)
    sent_A.append(s)
    frame_A.append(f)
    if (i + 1) % 20 == 0:
        print(f"  Prompt A: {i + 1}/200 rows done")
 
print("\n--- Running Prompt B (alternative) ---")
sent_B = []
frame_B = []
 
for i, row in sample.iterrows():
    text = str(row.get("video_description_english", ""))
    s = classify_text(text, SENTIMENT_PROMPT_B, VALID_SENTIMENT)
    f = classify_text(text, FRAME_PROMPT_B, VALID_FRAMES)
    sent_B.append(s)
    frame_B.append(f)
    if (i + 1) % 20 == 0:
        print(f"  Prompt B: {i + 1}/200 rows done")
 
# Save results
sample['sent_prompt_A'] = sent_A
sample['frame_prompt_A'] = frame_A
sample['sent_prompt_B'] = sent_B
sample['frame_prompt_B'] = frame_B
sample.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nResults saved to: {OUTPUT_FILE}")

from sklearn.metrics import cohen_kappa_score
 
print("\n" + "=" * 60)
print("  PROMPT SENSITIVITY RESULTS")
print("=" * 60)
 
total = len(sample)
 
# Sentiment agreement
sent_agree = (sample['sent_prompt_A'] == sample['sent_prompt_B']).sum()
sent_kappa  = cohen_kappa_score(sample['sent_prompt_A'], sample['sent_prompt_B'])
 
# Frame agreement
frame_agree = (sample['frame_prompt_A'] == sample['frame_prompt_B']).sum()
frame_kappa  = cohen_kappa_score(sample['frame_prompt_A'], sample['frame_prompt_B'])
 
print(f"\nSample size: {total} rows")
print(f"Model: {MODEL_NAME}")
print(f"\nTier 1 — Sentiment")
print(f"  Agreement between Prompt A and B : {sent_agree}/{total} ({sent_agree/total*100:.1f}%)")
print(f"  Cohen's Kappa (A vs B)           : {sent_kappa:.4f}")
 
print(f"\nTier 3 — Narrative Frame")
print(f"  Agreement between Prompt A and B : {frame_agree}/{total} ({frame_agree/total*100:.1f}%)")
print(f"  Cohen's Kappa (A vs B)           : {frame_kappa:.4f}")
 
# Label distribution comparison
print(f"\n--- Sentiment distribution comparison ---")
print(f"{'Label':<15} {'Prompt A':>10}  {'Prompt B':>10}")
print("-" * 40)
for label in VALID_SENTIMENT:
    a = sent_A.count(label)
    b = sent_B.count(label)
    print(f"  {label:<13} {a:>10}  {b:>10}")
 
print(f"\n--- Frame distribution comparison ---")
print(f"{'Label':<25} {'Prompt A':>10}  {'Prompt B':>10}")
print("-" * 50)
for label in VALID_FRAMES:
    a = frame_A.count(label)
    b = frame_B.count(label)
    print(f"  {label:<23} {a:>10}  {b:>10}")
 
# Rows where labels changed
sent_changed  = sample[sample['sent_prompt_A'] != sample['sent_prompt_B']]
frame_changed = sample[sample['frame_prompt_A'] != sample['frame_prompt_B']]
 
print(f"\n--- Rows where sentiment label changed ({len(sent_changed)}) ---")
for _, row in sent_changed.iterrows():
    print(f"  A: {row['sent_prompt_A']:<15} B: {row['sent_prompt_B']:<15} Text: {str(row['video_description_english'])[:60]}")
 
print(f"\n--- Rows where frame label changed ({len(frame_changed)}) ---")
for _, row in frame_changed.iterrows():
    print(f"  A: {row['frame_prompt_A']:<25} B: {row['frame_prompt_B']:<25} Text: {str(row['video_description_english'])[:50]}")
 
print("\n--- INTERPRETATION ---")
for tier, kappa in [("Sentiment", sent_kappa), ("Frame", frame_kappa)]:
    if kappa >= 0.81:   interp = "Almost perfect — classification is highly robust to prompt wording"
    elif kappa >= 0.61: interp = "Substantial — classification is largely robust to prompt wording"
    elif kappa >= 0.41: interp = "Moderate — some sensitivity to prompt wording exists"
    else:               interp = "Fair/Poor — classification is sensitive to prompt wording"
    print(f"  {tier}: κ={kappa:.3f} — {interp}")
