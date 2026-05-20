# Iran Crisis TikTok Data - Videos (Comments Belong To) Preprocessing Pipeline

import os
import re
import pandas as pd
from datetime import datetime

# CONFIGURATION
INPUT_FILE  = "data/raw/iran_crisis_videos_comments_belong_to.csv"
OUTPUT_FILE = "data/clean/iran_crisis_videos_belong_to_clean.csv"


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


# Drop rows with missing video id
# Without a valid id we cannot merge this file with comments
print("---Dropping rows with missing video id---")

before = len(df)
df = df.dropna(subset=["id"])
after  = len(df)
print(f"Removed {before - after} rows with missing id")
print(f"Remaining rows: {after}")


# Fix numeric columns
print("---Fixing numeric columns---")

numeric_cols = ["like_count", "share_count", "comment_count", "view_count"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
print(f"Fixed columns: {numeric_cols}")


# Fix region_code
# Some region_code values are corrupted timestamps instead of country codes
# Valid region codes are 2-letter country codes like US, GB, DE etc.
print("---Fixing region_code---")

def fix_region_code(code):
    if not isinstance(code, str):
        return None
    code = code.strip()
    # Valid region code is exactly 2 uppercase letters
    if re.match(r"^[A-Z]{2}$", code):
        return code
    return None

df["region_code"] = df["region_code"].apply(fix_region_code)
print(f"Valid region codes remaining: {df['region_code'].notna().sum()}")
print(f"Unique region codes: {df['region_code'].dropna().unique().tolist()}")

# Clean hashtag column
df["hashtag"] = df["hashtag"].apply(
    lambda x: re.sub(r"[\x00-\x1f\x7f-\x9f]", "", str(x)).strip()
    if isinstance(x, str) else x
)
df["hashtag"] = df["hashtag"].apply(
    lambda x: x if isinstance(x, str) and len(x) > 2 else None
)


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

# Remove rows where timestamp converted to 1970 (corrupted)
df = df[df["create_time_readable"] > "1970-01-02"]

# Clean video description
print("---Cleaning video description---")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    # Remove escape characters
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove hashtags
    text = re.sub(r"#\w+", "", text)
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["video_description_clean"] = df["video_description"].apply(clean_text)
print("Video descriptions cleaned")


# Deduplication
print("---Deduplication---")

before = len(df)
df = df.drop_duplicates(subset="id", keep="first")
after  = len(df)
print(f"Removed {before - after} duplicate rows")
print(f"Remaining rows: {after}")


# Final metadata tagging and column selection
print("---Final metadata tagging---")

KEEP_COLUMNS = [
    "id",
    "video_description_clean",
    "region_code",
    "like_count",
    "share_count",
    "comment_count",
    "view_count",
    "username",
    "hashtag",
    "hashtag_names",
    "date_range",
    "create_time_readable",
    "voice_to_text",
]

KEEP_COLUMNS = [c for c in KEEP_COLUMNS if c in df.columns]
df_clean     = df[KEEP_COLUMNS].copy()

# Rename id to video_id for clarity when merging with comments
df_clean = df_clean.rename(columns={"id": "video_id"})

print(f"Final columns: {df_clean.columns.tolist()}")


# Save clean output
print("---Saving clean dataset---")

os.makedirs("data/clean", exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Clean dataset saved to: {OUTPUT_FILE}")


# Final summary
print("---PREPROCESSING SUMMARY---")

print(f"Input rows          : 507")
print(f"Output rows         : {len(df_clean)}")
print(f"Unique hashtags     : {df_clean['hashtag'].dropna().unique().tolist()}")
print(f"Countries (region)  : {df_clean['region_code'].nunique()}")
print(f"Date range          : {df_clean['create_time_readable'].min()} "
      f"to {df_clean['create_time_readable'].max()}")