# Iran Crisis - Merge Videos Script
# Combines iran_crisis_videos_clean.csv and iran_crisis_videos2_clean.csv
# into one complete videos dataset


import pandas as pd
import os

# CONFIGURATION
FILE1       = "data/clean/iran_crisis_videos_clean.csv"
FILE2       = "data/clean/iran_crisis_videos2_clean.csv"
OUTPUT_FILE = "data/clean/iran_crisis_videos_all_clean.csv"


# Load both clean files
print("---Loading first clean videos file---")
df1 = pd.read_csv(FILE1, encoding="utf-8-sig")
print(f"File 1 rows: {len(df1)}")

print("---Loading second clean videos file---")
df2 = pd.read_csv(FILE2, encoding="utf-8-sig")
print(f"File 2 rows: {len(df2)}")


# Combine both files
print("---Combining both files---")
combined = pd.concat([df1, df2], ignore_index=True)
print(f"Combined rows before deduplication: {len(combined)}")


# Remove duplicates that may exist across both files
print("---Removing duplicates across both files---")
before = len(combined)
combined = combined.drop_duplicates(subset="id", keep="first")
combined = combined.drop_duplicates(subset="video_description_clean", keep="first")
after = len(combined)
print(f"Removed {before - after} duplicate rows")
print(f"Remaining rows: {after}")


# Save combined output
print("---Saving combined dataset---")
os.makedirs("data/clean", exist_ok=True)
combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Combined dataset saved to: {OUTPUT_FILE}")


# Summary
print("---MERGE SUMMARY---")
print(f"File 1 rows          : {len(df1)}")
print(f"File 2 rows          : {len(df2)}")
print(f"Combined rows        : {len(combined)}")
print(f"Languages detected   : {combined['language'].nunique()}")
print(f"Language breakdown   :")
print(combined["language_full"].value_counts().to_string())
print(f"Countries (region)   : {combined['region_code'].nunique()}")
print(f"Date range           : {combined['create_time_readable'].min()} "
      f"to {combined['create_time_readable'].max()}")
print(f"Hashtags present     : {combined['hashtag'].dropna().unique().tolist()}")
print(f"Output file          : {OUTPUT_FILE}")
