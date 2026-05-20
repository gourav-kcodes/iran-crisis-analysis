# Iran Crisis TikTok Data - Merge Script
#
# This script merges the clean comments file with the clean videos metadata file
# so each comment gets enriched with its parent video's region, hashtag, date etc.

import os
import pandas as pd

# CONFIGURATION
COMMENTS_FILE  = "data/clean/iran_crisis_comments_clean.csv"
VIDEOS_FILE    = "data/clean/iran_crisis_videos_belong_to_clean.csv"
OUTPUT_FILE    = "data/clean/iran_crisis_merged.csv"


# Load clean comments
print("---Loading clean comments---")

comments = pd.read_csv(COMMENTS_FILE, encoding="utf-8-sig")
print(f"Comments rows    : {len(comments)}")
print(f"Comments columns : {comments.columns.tolist()}")


# Load clean videos metadata
print("\n---Loading clean videos metadata---")

videos = pd.read_csv(VIDEOS_FILE, encoding="utf-8-sig")
print(f"Videos rows      : {len(videos)}")
print(f"Videos columns   : {videos.columns.tolist()}")


# Fix video_id format for merging
# Both files store video_id as float strings due to precision loss
# We need to convert both to string for reliable matching
print("\n---Fixing video_id format for merging---")

def normalize_video_id(val):
    try:
        # Convert to float first then to int then to string
        # This removes the scientific notation and decimal
        return str(int(float(val)))
    except:
        return None

comments["video_id_str"] = comments["video_id"].apply(normalize_video_id)
videos["video_id_str"]   = videos["video_id"].apply(normalize_video_id)

print(f"Sample comment video_id_str : {comments['video_id_str'].head(3).tolist()}")
print(f"Sample video video_id_str   : {videos['video_id_str'].head(3).tolist()}")


# Rename video columns before merging to avoid conflicts
# Both files may have like_count, username etc. — prefix video columns with video_
video_rename = {
    "video_description_clean" : "video_description",
    "like_count"              : "video_like_count",
    "share_count"             : "video_share_count",
    "comment_count"           : "video_comment_count",
    "view_count"              : "video_view_count",
    "username"                : "video_username",
    "create_time_readable"    : "video_create_time",
    "voice_to_text"           : "video_voice_to_text",
}

videos = videos.rename(columns=video_rename)


# Merge comments with videos metadata on video_id_str
print("\n---Merging comments with video metadata---")

merged = pd.merge(
    comments,
    videos,
    on      = "video_id_str",
    how     = "left"    # keep all comments even if video metadata is missing
)

print(f"Merged rows      : {len(merged)}")
print(f"Merged columns   : {merged.columns.tolist()}")

# Check how many comments got matched with video metadata
matched = merged["region_code"].notna().sum()
print(f"Comments matched with video metadata: {matched} out of {len(merged)}")


# Drop helper columns no longer needed
merged = merged.drop(columns=["video_id_str", "video_id_x", "video_id_y"],
                     errors="ignore")


# Final column selection — keep everything useful
KEEP_COLUMNS = [
    "id",                    # comment id
    "text_clean",            # cleaned comment text
    "text_english",          # translated comment text
    "language",
    "language_full",
    "create_time_readable",  # when comment was posted
    "like_count",            # comment likes
    "reply_count",           # comment replies
    "parent_comment_id",     # if reply, which comment it replied to
    "platform",              # TikTok
    "video_id",              # which video this comment belongs to
    "video_description",     # parent video description
    "region_code",           # country of video poster
    "hashtag",               # which hashtag the video was collected under
    "hashtag_names",         # all hashtags in the video
    "date_range",            # collection window
    "video_like_count",      # parent video likes
    "video_view_count",      # parent video views
    "video_comment_count",   # parent video comment count
    "video_share_count",     # parent video shares
    "video_username",        # parent video poster
    "video_create_time",     # when parent video was posted
]

KEEP_COLUMNS = [c for c in KEEP_COLUMNS if c in merged.columns]
df_final     = merged[KEEP_COLUMNS].copy()
print(f"\nFinal columns: {df_final.columns.tolist()}")


# Save merged output
print("\n---Saving merged dataset---")

os.makedirs("data/clean", exist_ok=True)
df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Merged dataset saved to: {OUTPUT_FILE}")


# Final summary
print("\n---MERGE SUMMARY---")

print(f"Total comments          : {len(df_final)}")
print(f"Comments with region    : {df_final['region_code'].notna().sum()}")
print(f"Comments with hashtag   : {df_final['hashtag'].notna().sum()}")
print(f"Languages detected      : {df_final['language'].nunique()}")
print(f"Countries represented   : {df_final['region_code'].nunique()}")
print(f"Unique hashtags         : {df_final['hashtag'].dropna().unique().tolist()}")
print(f"Date range              : {df_final['create_time_readable'].min()} "
      f"to {df_final['create_time_readable'].max()}")
print(f"Platform                : TikTok")