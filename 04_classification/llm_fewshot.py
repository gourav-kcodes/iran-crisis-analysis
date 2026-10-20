# Few-shot classification of the video descriptions with an open-source LLM (run on a Kaggle T4 GPU).
# Exported from the Kaggle notebooks used in the project (Mistral 7B, Qwen2.5-3B and DeepSeek-R1 few-shot runs);
# Kaggle paths were replaced by command-line arguments and the test cells were dropped. Prompts, examples,
# decoding settings and the short-text rule are unchanged.
#
# The few-shot examples (2 per label) are read from few_shot_examples.py in this folder. That file is NOT in the
# repository because most examples are real TikTok descriptions; copy few_shot_examples_TEMPLATE.py to
# few_shot_examples.py and fill it in to run this script.
#
# Usage (project root):
#   python 04_classification/llm_fewshot.py mistral  data/clean/iran_crisis_videos_all_clean.csv data/clean/iran_crisis_videos_classified_mistral_fewshot.csv
#   python 04_classification/llm_fewshot.py qwen     <input.csv> <output.csv>
#   python 04_classification/llm_fewshot.py deepseek <input.csv> <output.csv>
# The run saves every 100 rows and resumes from the output file if it already exists.
import os, re, sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from few_shot_examples import FEW_SHOT_SENTIMENT, FEW_SHOT_FRAME, FEAR_EXAMPLES_QWEN

MODELS = {'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
          'qwen': 'Qwen/Qwen2.5-3B-Instruct',
          'deepseek': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'}
KEY, INPUT_FILE, OUTPUT_FILE = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL_NAME = MODELS[KEY]
IS_DEEPSEEK = KEY == 'deepseek'
MAX_NEW_TOKENS = 500 if IS_DEEPSEEK else 20      # DeepSeek needs room for its <think> block
SAVE_EVERY = 100

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=IS_DEEPSEEK)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', torch_dtype=torch.float16,
                                             trust_remote_code=IS_DEEPSEEK)
model.eval()

# Mistral and DeepSeek used the same examples; Qwen used two differently worded Fear examples.
sentiment_examples = dict(FEW_SHOT_SENTIMENT)
if KEY == 'qwen':
    sentiment_examples['Fear'] = FEAR_EXAMPLES_QWEN


def build_few_shot_block(examples_dict):
    block = "\nHere are examples of correctly labeled texts:\n"
    for label, examples in examples_dict.items():
        for ex in examples:
            block += f'\nText: "{ex}"\nLabel: {label}\n'
    return block


SENTIMENT_EXAMPLES_BLOCK = build_few_shot_block(sentiment_examples)
FRAME_EXAMPLES_BLOCK = build_few_shot_block(FEW_SHOT_FRAME)

VALID_SENTIMENT = ["Fear", "Anger", "Sadness", "Hope", "Solidarity",
                   "Frustration", "Humor", "Disgust", "Curiosity", "Neutral"]
VALID_FRAMES = ["Humanitarian", "Political blame", "Military/strategic",
                "Personal connection", "Economic/sanctions", "Historical context",
                "Religious framing", "Media criticism", "Misinformation",
                "Satire/meme", "Unclear"]

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
""" + SENTIMENT_EXAMPLES_BLOCK + """
Now classify this new text:
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
""" + FRAME_EXAMPLES_BLOCK + """
Now classify this new text:
Text: "[TEXT HERE]"
 
Respond with ONLY the label. Nothing else."""


def classify_text(text, prompt_template, valid_labels, min_words=4, min_chars=10):
    if not isinstance(text, str):
        return "Unclear"
    text_clean = text.strip()
    if len(text_clean) < min_chars or len(text_clean.split()) < min_words:     # short texts are not sent to the model
        return "Unclear"
    try:
        prompt = prompt_template.replace("[TEXT HERE]", text_clean[:1000])
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True,
                                                add_generation_prompt=True)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        if IS_DEEPSEEK:
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
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
    result_row["model_used"] = MODEL_NAME + "_fewshot"
    results.append(result_row)
    if (i + 1) % 50 == 0 or (i + 1) == total:
        print(f"  Processed {i + 1}/{total} rows")
    if (i + 1) % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Saved {len(results)} rows to {OUTPUT_FILE}")
