# Iran Crisis - TikTok Data Collection Script

# Requirements: pip install researchtikpy pandas
# Replace CLIENT_KEY and CLIENT_SECRET with your actual credentials

import researchtikpy as rtk
import pandas as pd

# API Credentials
CLIENT_KEY    = "your_key_here"
CLIENT_SECRET = "your_secret_here"

# Generate Access Token
print("Generating access token...")
access_token_dict = rtk.get_access_token(CLIENT_KEY, CLIENT_SECRET)
access_token      = access_token_dict["access_token"]
print("Access token obtained.")

# Search Parameters
HASHTAGS = [
    "IranCrisis",
    "IranWar",
    "IranNuclear",
    "Iran",
    "IranProtest"
]

# NOTE: TikTok Research API requires date range within 30 days per request.
# We split into two 30-day windows to cover March 1 - April 30, 2026.
DATE_RANGES = [
    ("20260301", "20260331"),  # March 2026
    ("20260401", "20260430"),  # April 2026
]

MAX_VIDEOS_PER_HASHTAG_PER_WINDOW = 100  # adjust as needed

# Step 1: Collect Videos
all_videos = []

for start_date, end_date in DATE_RANGES:
    print(f"\nCollecting videos: {start_date} to {end_date}")
    for hashtag in HASHTAGS:
        print(f"  Hashtag: #{hashtag} ...")
        try:
            videos_df = rtk.get_videos_hashtag(
                hashtags        = [hashtag],
                access_token    = access_token,
                start_date      = start_date,
                end_date        = end_date,
                total_max_count = MAX_VIDEOS_PER_HASHTAG_PER_WINDOW
            )
            videos_df["hashtag"]    = hashtag
            videos_df["date_range"] = f"{start_date}_{end_date}"
            all_videos.append(videos_df)
            print(f"  -> {len(videos_df)} videos collected")
        except Exception as e:
            print(f"  -> Error for #{hashtag}: {e}")

# Combine and deduplicate
videos_combined = pd.concat(all_videos, ignore_index=True)
videos_combined = videos_combined.drop_duplicates(subset="id")
print(f"\nTotal unique videos collected: {len(videos_combined)}")

# Save
videos_combined.to_csv("iran_crisis_videos.csv", index=False)
print("Saved to iran_crisis_videos.csv")

# Step 2: Collect Comments
print("\nCollecting comments for all videos...")
try:
    comments_df = rtk.get_video_comments(
        videos_df    = videos_combined,
        access_token = access_token,
        max_count    = 50    # max comments per video
    )
    comments_df = comments_df.drop_duplicates(subset="id")
    print(f"Total unique comments collected: {len(comments_df)}")

    # Save
    comments_df.to_csv("iran_crisis_comments.csv", index=False)
    print("Saved to iran_crisis_comments.csv")

except Exception as e:
    print(f"Error collecting comments: {e}")

# Summary
print("\n── Collection Summary ──────────────────")
print(f"Hashtags searched : {HASHTAGS}")
print(f"Date windows      : {DATE_RANGES}")
print(f"Total videos      : {len(videos_combined)}")
try:
    print(f"Total comments    : {len(comments_df)}")
except:
    print("Total comments    : see error above")
print("───────────")