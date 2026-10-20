# Zero-shot classification of the video descriptions with Mistral 7B or Qwen2.5-3B (run on a Kaggle T4 GPU).
# The original zero-shot script was not kept. This file is rebuilt from the project notebooks: the prompts are
# "Prompt A" of the prompt-sensitivity notebook (05_evaluation/prompt_sensitivity.py), which that notebook marks as
# the prompt used in zero-shot classification, and the classify function and decoding settings are the same as there.
#
# Usage (project root):
#   python 04_classification/llm_zeroshot.py mistral data/clean/iran_crisis_videos_all_clean.csv data/clean/iran_crisis_videos_classified_mistral.csv
#   python 04_classification/llm_zeroshot.py qwen    data/clean/iran_crisis_videos_all_clean.csv data/clean/iran_crisis_videos_classified_qwen.csv
# The run saves every 100 rows and resumes from the output file if it already exists.
import os, sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {'mistral': 'mistralai/Mistral-7B-Instruct-v0.3', 'qwen': 'Qwen/Qwen2.5-3B-Instruct'}
MODEL_NAME = MODELS[sys.argv[1]]
INPUT_FILE, OUTPUT_FILE = sys.argv[2], sys.argv[3]
SAVE_EVERY = 100

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
model.eval()

SENTIMENT_PROMPT = """You are a social science researcher analyzing public reactions to the Iran crisis on TikTok.
 
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

FRAME_PROMPT = """You are a social science researcher analyzing public reactions to the Iran crisis on TikTok.
 
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
if os.path.exists(OUTPUT_FILE):
    df_existing = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
    start_idx = len(df_existing)
    results = df_existing.to_dict("records")
    print(f"Resuming from row {start_idx}")
else:
    start_idx, results = 0, []
df_remaining = df.iloc[start_idx:].reset_index(drop=True)
total = len(df_remaining)
for i, row in df_remaining.iterrows():
    text = str(row.get("video_description_english", ""))
    result_row = row.to_dict()
    result_row["sentiment_label"] = classify_text(text, SENTIMENT_PROMPT, VALID_SENTIMENT)
    result_row["frame_label"] = classify_text(text, FRAME_PROMPT, VALID_FRAMES)
    result_row["model_used"] = MODEL_NAME
    results.append(result_row)
    if (i + 1) % 50 == 0 or (i + 1) == total:
        print(f"  Processed {i + 1}/{total} rows")
    if (i + 1) % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Saved {len(results)} rows to {OUTPUT_FILE}")
