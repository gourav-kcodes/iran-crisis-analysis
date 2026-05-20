# Iran Crisis TikTok Data - Preprocessing Pipeline

import os
import re
import emoji
import pandas as pd
from datetime import datetime
from langdetect import detect, LangDetectException
from transformers import MarianMTModel, MarianTokenizer

# CONFIGURATION — change paths here if needed
INPUT_FILE  = "data/raw/iran_crisis_videos.csv"
OUTPUT_FILE = "data/clean/iran_crisis_videos_clean.csv"
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


# Fix numeric columns
print("---Fixing numeric columns---")

numeric_cols = ["like_count", "share_count", "comment_count", "view_count"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
print(f"Fixed columns: {numeric_cols}")


# Convert timestamp
print("---Converting timestamps---")

def convert_timestamp(ts):
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None

df["create_time_readable"] = df["create_time"].apply(convert_timestamp)
print("Timestamps converted to readable format (UTC)")
print(f"Sample: {df['create_time_readable'].head(3).tolist()}")


# Clean video descriptions
print("---Cleaning video descriptions---")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove escape characters
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove emojis
    text = emoji.replace_emoji(text, replace="")
    # Remove hashtags (keep words, remove # symbol and tag)
    text = re.sub(r"#\w+", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["video_description_clean"] = df["video_description"].apply(clean_text)
print("Descriptions cleaned — URLs removed, whitespace normalized")

# Clean hashtag column — remove corrupted values
print("---Cleaning hashtag column---")
df["hashtag"] = df["hashtag"].apply(
    lambda x: re.sub(r"[\x00-\x1f\x7f-\x9f]", "", str(x)).strip()
    if isinstance(x, str) else x
)
df["hashtag"] = df["hashtag"].apply(
    lambda x: x if isinstance(x, str) 
    and re.match(r"^[A-Za-z0-9]+$", x) 
    and len(x) > 2 
    else None
)
print(f"Clean hashtags: {df['hashtag'].dropna().unique().tolist()}")

# Deduplication
print("---Deduplication---")

before = len(df)
df = df.drop_duplicates(subset="id", keep="first")
df = df.drop_duplicates(subset="video_description_clean", keep="first")
after  = len(df)
print(f"Removed {before - after} duplicate rows")
print(f"Remaining rows: {after}")


# Minimum length filtering
print("---Minimum length filtering (< 5 words removed)---")

before = len(df)
df = df[df["video_description_clean"].apply(
    lambda x: len(x.split()) >= MIN_WORD_COUNT
)]
after = len(df)
print(f"Removed {before - after} rows with fewer than {MIN_WORD_COUNT} words")
print(f"Remaining rows: {after}")


# Spam filtering
print("---Spam and noise filtering---")

SPAM_PATTERNS = [
    r"^[#\s]+$",          # only hashtags
    r"^[\W\d\s]+$",       # only symbols/numbers
    r"follow me",
    r"click the link",
    r"check bio",
    r"dm for",
    r"link in bio",       # very common TikTok spam
    r"subscribe",         # promotional
    r"giveaway",          # giveaway spam
    r"promo code",        # promotional
    r"discount",          # promotional
    r"shop now",          # promotional
    r"buy now",           # promotional
    r"onlyfans",          # irrelevant content
]

def is_spam(text):
    if not isinstance(text, str):
        return True
    text_lower = text.lower()
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

before = len(df)
df = df[~df["video_description_clean"].apply(is_spam)]
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

df["language"] = df["video_description_clean"].apply(detect_language)

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

df["video_description_english"] = df.apply(
    lambda row: translate_to_english(
        row["video_description_clean"],
        row["language"]
    ), axis=1
)

print("Translation complete.")
print("Original text preserved in : 'video_description_clean'")
print("English text stored in     : 'video_description_english'")


# Final metadata tagging and column selection
print("---Final metadata tagging---")

KEEP_COLUMNS = [
    "id",
    "video_description_clean",
    "video_description_english",
    "language",
    "language_full",
    "create_time_readable",
    "region_code",
    "like_count",
    "share_count",
    "comment_count",
    "view_count",
    "username",
    "hashtag",
    "hashtag_names",
    "date_range",
    "voice_to_text"
]

KEEP_COLUMNS       = [c for c in KEEP_COLUMNS if c in df.columns]
df_clean           = df[KEEP_COLUMNS].copy()
df_clean["platform"] = "TikTok"
print(f"Final columns: {df_clean.columns.tolist()}")

# Save clean output
print("---Saving clean dataset---")

os.makedirs("data/clean", exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Clean dataset saved to: {OUTPUT_FILE}")

# FINAL SUMMARY
print("---PREPROCESSING SUMMARY---")

print(f"Output rows         : {len(df_clean)}")
print(f"Languages detected  : {df_clean['language'].nunique()}")
print(f"Language breakdown  :")
print(df_clean["language"].value_counts().to_string())
print(f"Countries (region)  : {df_clean['region_code'].nunique()}")
print(f"Date range          : {df_clean['create_time_readable'].min()} "
      f"to {df_clean['create_time_readable'].max()}")
print(f"Hashtags present    : {df_clean['hashtag'].dropna().unique().tolist()}")
print(f"Platform            : TikTok")