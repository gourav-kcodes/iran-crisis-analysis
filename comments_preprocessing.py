# Iran Crisis TikTok Data - Comments Preprocessing Pipeline

import os
import re
import emoji
import pandas as pd
from datetime import datetime
from langdetect import detect, LangDetectException
from transformers import MarianMTModel, MarianTokenizer

# CONFIGURATION
INPUT_FILE  = "data/raw/iran_crisis_comments.csv"
OUTPUT_FILE = "data/clean/iran_crisis_comments_clean.csv"
MIN_WORD_COUNT     = 3
MULTILINGUAL_MODEL = "Helsinki-NLP/opus-mt-mul-en"


# Load the data
print("---Loading data---")

df = pd.read_csv(
    INPUT_FILE,
    encoding     = "unicode_escape",
    on_bad_lines = "skip",
    engine       = "python"
)
df.columns = df.columns.str.replace("ï»¿", "").str.strip()
print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")


# Drop rows with missing text
print("---Dropping rows with missing text---")

before = len(df)
df = df.dropna(subset=["text"])
after  = len(df)
print(f"Removed {before - after} rows with missing text")
print(f"Remaining rows: {after}")


# Fix numeric columns
print("---Fixing numeric columns---")

# like_count has anomalous huge values (likely corrupted IDs stored as floats)
# cap like_count at a reasonable maximum of 100000
df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
df["like_count"] = df["like_count"].apply(lambda x: x if x <= 100000 else 0).astype(int)

# reply_count — straightforward
df["reply_count"] = pd.to_numeric(df["reply_count"], errors="coerce").fillna(0).astype(int)

print("Fixed columns: like_count (capped at 100000), reply_count")


# Convert timestamp
print("---Converting timestamps---")

def convert_timestamp(ts):
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None

df["create_time_readable"] = df["create_time"].apply(convert_timestamp)
print("Timestamps converted to readable format (UTC)")
print(f"Sample: {df['create_time_readable'].dropna().head(3).tolist()}")


# Clean comment text
print("---Cleaning comment text---")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove control characters (very common in TikTok comments - \x0f, \x19 etc)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    # Remove escape characters
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove emojis
    text = emoji.replace_emoji(text, replace="")
    # Remove hashtags
    text = re.sub(r"#\w+", "", text)
    # Remove mentions (@username)
    text = re.sub(r"@\w+", "", text)
    # Remove excessive punctuation (e.g. !!!!! or -----)
    text = re.sub(r"([^\w\s])\1{2,}", "", text)
    # Remove excessive repeated characters (e.g. ppppppp or 333333)
    text = re.sub(r"(.)\1{3,}", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_clean"] = df["text"].apply(clean_text)
print("Comments cleaned — control characters, URLs, emojis, mentions removed")


# Deduplication
print("---Deduplication---")

before = len(df)
df = df.drop_duplicates(subset="id", keep="first")
df = df.drop_duplicates(subset="text_clean", keep="first")
after  = len(df)
print(f"Removed {before - after} duplicate rows")
print(f"Remaining rows: {after}")


# Minimum length filtering
print("---Minimum length filtering (< 3 words removed)---")

before = len(df)
df = df[df["text_clean"].apply(lambda x: len(x.split()) >= MIN_WORD_COUNT)]
after  = len(df)
print(f"Removed {before - after} rows with fewer than {MIN_WORD_COUNT} words")
print(f"Remaining rows: {after}")


# Spam filtering
print("---Spam and noise filtering---")

SPAM_PATTERNS = [
    r"^\s*$",           # empty or whitespace only after cleaning
    r"^[#\s]+$",        # only hashtags
    r"^[\W\d\s]+$",     # only symbols or numbers
    r"^p+$",            # only p characters
    r"^m+$",            # only m characters
    r"^d+$",            # only d characters
    r"follow me",
    r"click the link",
    r"check bio",
    r"link in bio",
    r"dm for",
    r"subscribe",
    r"giveaway",
    r"promo code",
    r"discount",
    r"shop now",
    r"buy now",
    r"onlyfans",
]

def is_spam(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return True
    text_lower = text.lower()
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

before = len(df)
df = df[~df["text_clean"].apply(is_spam)]
after  = len(df)
print(f"Removed {before - after} spam/noise rows")
print(f"Remaining rows: {after}")


# Language detection
print("---Language detection---")

def detect_language(text):
    try:
        if not isinstance(text, str) or len(text.strip()) < 10:
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"

df["language"] = df["text_clean"].apply(detect_language)

# Map language codes to full names
LANGUAGE_NAMES = {
    "en": "English", "de": "German", "id": "Indonesian",
    "es": "Spanish", "fr": "French", "tl": "Tagalog",
    "so": "Somali", "vi": "Vietnamese", "ca": "Catalan",
    "pt": "Portuguese", "it": "Italian", "cy": "Welsh",
    "da": "Danish", "tr": "Turkish", "hr": "Croatian",
    "et": "Estonian", "hu": "Hungarian", "sw": "Swahili",
    "no": "Norwegian", "ro": "Romanian", "sq": "Albanian",
    "lv": "Latvian", "nl": "Dutch", "af": "Afrikaans",
    "fi": "Finnish", "pl": "Polish", "sv": "Swedish",
    "sl": "Slovenian", "sk": "Slovak", "cs": "Czech",
    "ar": "Arabic", "fa": "Persian", "hi": "Hindi",
    "ur": "Urdu", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "unknown": "Unknown",
    " ru": "Russian",
    "ru": "Russian",
    "zh": "Chinese", 
    "ar": "Arabic",
    "fa": "Persian",
    "hi": "Hindi",
    "ur": "Urdu",
    "ja": "Japanese",
    "ko": "Korean",
    "bn": "Bengali",
    "th": "Thai",
    "he": "Hebrew",
    "mk": "Macedonian",
    "bg": "Bulgarian",
    "uk": "Ukrainian",
    "az": "Azerbaijani",
    "ms": "Malay",
    "gl": "Galician",
}
df["language_full"] = df["language"].map(LANGUAGE_NAMES).fillna(df["language"])

print("Language distribution:")
print(df["language"].value_counts().to_string())


# Translation to English
print("---Translation to English---")

print(f"Loading multilingual model: {MULTILINGUAL_MODEL}")
print("(First run downloads the model — cached after that)")

tokenizer_mul = MarianTokenizer.from_pretrained(MULTILINGUAL_MODEL)
model_mul     = MarianMTModel.from_pretrained(MULTILINGUAL_MODEL)
print("Model loaded successfully.")

def translate_to_english(text, lang):
    if lang in ["en", "unknown"]:
        return text
    try:
        tokens = tokenizer_mul(
            [text],
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = 512
        )
        translated = model_mul.generate(**tokens)
        return tokenizer_mul.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"  Translation error for lang '{lang}': {e}")
        return text

non_english_count = (df["language"] != "en").sum()
print(f"Translating {non_english_count} non-English rows...")

df["text_english"] = df.apply(
    lambda row: translate_to_english(
        row["text_clean"],
        row["language"]
    ), axis=1
)

print("Translation complete.")
print("Original text preserved in : 'text_clean'")
print("English text stored in     : 'text_english'")


# Final metadata tagging and column selection
print("---Final metadata tagging---")

KEEP_COLUMNS = [
    "id",
    "video_id",
    "text_clean",
    "text_english",
    "language",
    "language_full",
    "create_time_readable",
    "like_count",
    "reply_count",
    "parent_comment_id",
]

KEEP_COLUMNS         = [c for c in KEEP_COLUMNS if c in df.columns]
df_clean             = df[KEEP_COLUMNS].copy()
df_clean["platform"] = "TikTok"
print(f"Final columns: {df_clean.columns.tolist()}")


# Save clean output
print("---Saving clean dataset---")

os.makedirs("data/clean", exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Clean dataset saved to: {OUTPUT_FILE}")


# Final summary
print("---PREPROCESSING SUMMARY---")

print(f"Input rows          : 3611")
print(f"Output rows         : {len(df_clean)}")
print(f"Languages detected  : {df_clean['language'].nunique()}")
print(f"Language breakdown  :")
print(df_clean["language"].value_counts().to_string())
print(f"Unique videos       : {df_clean['video_id'].nunique()}")
print(f"Date range          : {df_clean['create_time_readable'].min()} "
      f"to {df_clean['create_time_readable'].max()}")
print(f"Platform            : TikTok")