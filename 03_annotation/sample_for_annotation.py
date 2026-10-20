import pandas as pd
import os

df = pd.read_csv("data/clean/iran_crisis_videos_all_clean.csv", encoding="utf-8-sig")
print(f"Loaded {len(df)} rows")

# Simple random sample of 400 rows
sample = df.sample(n=400, random_state=42).reset_index(drop=True)

# Keep only what you need for annotation
annotation_df = sample[[
    "id",
    "video_description_clean",
    "video_description_english",
    "language_full",
    "hashtag",
    "create_time_readable",
    "region_code"
]].reset_index(drop=True)

# Add empty columns for annotation
annotation_df["tier1_sentiment"] = ""
annotation_df["tier3_frame"]     = ""
annotation_df["notes"]           = ""

# Save
os.makedirs("data/annotation", exist_ok=True)
annotation_df.to_csv("data/annotation/annotation_sample_400.csv", index=False, encoding="utf-8-sig")
print(f"Saved {len(annotation_df)} rows for annotation")
print(f"Hashtag distribution:\n{annotation_df['hashtag'].value_counts()}")