# Template for the few-shot examples used by llm_fewshot.py.
# The real file (few_shot_examples.py, same folder) is kept out of the repository because most of the examples are
# real TikTok descriptions, 29 of the 42 taken from gold-standard rows. Copy this file to few_shot_examples.py and
# put two example texts under each label.

FEW_SHOT_SENTIMENT = {          # Tier 1, 2 examples per label (Mistral and DeepSeek)
    "Fear": ["", ""], "Anger": ["", ""], "Sadness": ["", ""], "Hope": ["", ""], "Solidarity": ["", ""],
    "Frustration": ["", ""], "Humor": ["", ""], "Disgust": ["", ""], "Curiosity": ["", ""], "Neutral": ["", ""],
}

FEAR_EXAMPLES_QWEN = ["", ""]   # Qwen used two differently worded Fear examples; all other examples are shared

FEW_SHOT_FRAME = {              # Tier 3, 2 examples per label (all three models)
    "Humanitarian": ["", ""], "Political blame": ["", ""], "Military/strategic": ["", ""],
    "Personal connection": ["", ""], "Economic/sanctions": ["", ""], "Historical context": ["", ""],
    "Religious framing": ["", ""], "Media criticism": ["", ""], "Misinformation": ["", ""],
    "Satire/meme": ["", ""], "Unclear": ["", ""],
}
