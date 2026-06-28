"""
Scrapes TARGET_AI_SUBREDDITS and TARGET_REAL_SUBREDDITS, saves images,
and records metadata for each image in a CSV for later parquet packaging.

AI subreddits -> label: "fake"
Real subreddits -> label: "real"

Default mode: resumes from the last scraped post per subreddit (reads reddit_metadata.csv).
If a subreddit has never been scraped, goes back MAX_DAYS_AGO days.
Override with --days-ago N to force a fixed lookback for all subreddits.
"""

import praw, os, requests, csv, fcntl, argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import pandas as pd
from dotenv import load_dotenv
import subprocess
import sys
import cv2

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_DAYS_AGO = 1000 

TARGET_AI_SUBREDDITS = [
    "aigeneratedart", "aiArt", "midjourney", "aiimages", "AiArtwork", "AiGeneratedArt",
    "Pro_Ai_Art", "AI_ART", "aivideo", "AIVideos_SFW", "GenAIGallery", "deepdream",
    "nanobananaSFW", "VEO3", "Seedance_A",
    "KlingAI_Videos", "GrokImage",
]

TARGET_REAL_SUBREDDITS = [
    "photography", "itookapicture", "pics", "EarthPorn", "CityPorn",
    "FoodPorn", "StreetPhotography", "analog", "photocritique",
    "WildlifePhotography", "MacroPorn", "SkyPorn", "portraitphotography", "MODELING",
    "AmateurPhotography" , "portraitphotography", "portraits", "casual_photography",
    "portraitphotos"
]

REDDIT_STAGING_DIR = "data/reddit_images"
METADATA_CSV = os.path.join(REDDIT_STAGING_DIR, "reddit_metadata.csv")
METADATA_FIELDS = ["filename", "label", "subreddit", "post_date", "reddit_id", "packaged"]

NUM_OF_VIDEO_FRAMES = 15
TODAY = datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days-ago", type=int, default=None,
        help="Force a fixed lookback in days for all subreddits. "
            "Default: resume from last scraped post per subreddit."
    )
    parser.add_argument("--staging-dir", default=REDDIT_STAGING_DIR)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--creds-csv", default="data/creds/creds.csv")
    parser.add_argument("--skip-nsfw-filter", action="store_true")
    parser.add_argument("--nsfw-script", default="submit_nsfw_filtering.sh")
    parser.add_argument("--slurm-log-dir", default="data/slurm_logs")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------
def get_last_scraped_dates():
    """
    Read reddit_metadata.csv and return the most recent post_date per subreddit.
    Returns dict: subreddit (lowercase) -> datetime (UTC).
    """
    last_dates = {}
    if not os.path.exists(METADATA_CSV):
        return last_dates

    with open(METADATA_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row["subreddit"].lower()
            try:
                post_date = datetime.strptime(row["post_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if sub not in last_dates or post_date > last_dates[sub]:
                    last_dates[sub] = post_date
            except (ValueError, KeyError):
                continue

    return last_dates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def append_metadata(row_dict):
    """Append a metadata row to the shared CSV with file locking."""
    file_exists = os.path.exists(METADATA_CSV) and os.path.getsize(METADATA_CSV) > 0
    with open(METADATA_CSV, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
        fcntl.flock(f, fcntl.LOCK_UN)


def download_file(url, filepath):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        return False
    except:
        return False


def extract_video_frames(video_path, output_dir, base_filename, num_frames=NUM_OF_VIDEO_FRAMES):
    """Extracts a fixed number of frames evenly distributed across a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_frames = []

    if total_frames > 0:
        step_size = max(1, total_frames // num_frames)

        for i in range(num_frames):
            frame_idx = i * step_size
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_filename = f"{base_filename}_frame_{i}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_filename)
                print(f"Saved Video Frame: {frame_filename}")

    cap.release()
    return saved_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_reddit_scraper(days_ago=None):
    args_parsed = parse_args()
    global REDDIT_STAGING_DIR, METADATA_CSV
    REDDIT_STAGING_DIR = args_parsed.staging_dir
    METADATA_CSV = args_parsed.metadata_csv or os.path.join(REDDIT_STAGING_DIR, "reddit_metadata.csv")
    if days_ago is None:
        days_ago = args_parsed.days_ago

    credsfile = args_parsed.creds_csv
    cdf = pd.read_csv(credsfile, sep=",")
    creds = cdf[(cdf["datatype"] == "submissions")].to_dict(orient="records")[0]

    reddit = praw.Reddit(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
        user_agent=creds["useragent"],
    )

    os.makedirs(REDDIT_STAGING_DIR, exist_ok=True)

    # Determine cutoffs
    if days_ago is not None:
        # Fixed lookback mode
        global_cutoff = (datetime.now(timezone.utc) - timedelta(days=days_ago)).timestamp()
        per_sub_cutoffs = None
        print(f"Fixed lookback mode: scraping last {days_ago} days for all subreddits.")
    else:
        # Resume mode: per-subreddit cutoffs from metadata
        global_cutoff = (datetime.now(timezone.utc) - timedelta(days=MAX_DAYS_AGO)).timestamp()
        last_dates = get_last_scraped_dates()
        per_sub_cutoffs = {}
        for sub in [s.lower() for s in TARGET_AI_SUBREDDITS + TARGET_REAL_SUBREDDITS]:
            if sub in last_dates:
                per_sub_cutoffs[sub] = last_dates[sub].timestamp()
            else:
                per_sub_cutoffs[sub] = global_cutoff
        
        resumed = sum(1 for s in per_sub_cutoffs if s in last_dates)
        fresh = len(per_sub_cutoffs) - resumed
        print(f"Resume mode: {resumed} subreddits resuming from last post, {fresh} starting from {MAX_DAYS_AGO} days ago.")

    # Load existing filenames to skip duplicates
    existing_files = set()
    if os.path.exists(METADATA_CSV):
        with open(METADATA_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_files.add(row["filename"])
    print(f"Existing files in metadata: {len(existing_files)}")

    # Build combined list with labels
    subreddit_list = [(sub, "fake") for sub in TARGET_AI_SUBREDDITS] + \
                     [(sub, "real") for sub in TARGET_REAL_SUBREDDITS]

    failed_subreddits = []

    for sub_name, label in subreddit_list:
        try:
            sub_lower = sub_name.lower()

            if per_sub_cutoffs is not None:
                cutoff = per_sub_cutoffs[sub_lower]
                cutoff_date = datetime.fromtimestamp(cutoff, tz=timezone.utc).strftime("%Y-%m-%d")
                print(f"Scanning r/{sub_name} (label={label}, resuming from {cutoff_date})...")
            else:
                cutoff = global_cutoff
                print(f"Scanning r/{sub_name} (label={label})...")

            new_count = 0
            skipped_count = 0

            for submission in reddit.subreddit(sub_name).new(limit=None):
                if submission.created_utc < cutoff:
                    break

                date_str = datetime.fromtimestamp(
                    submission.created_utc, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                base_name = f"reddit_{sub_lower}_{date_str}_{submission.id}"
                url = str(submission.url).lower()

                # Handle images
                if url.endswith((".jpg", ".jpeg", ".png")):
                    file_extension = os.path.splitext(submission.url)[1]
                    file_name = f"{base_name}{file_extension}"

                    if file_name in existing_files:
                        skipped_count += 1
                        continue

                    filepath = os.path.join(REDDIT_STAGING_DIR, file_name)

                    if download_file(submission.url, filepath):
                        new_count += 1
                        append_metadata({
                            "filename": file_name,
                            "label": label,
                            "subreddit": sub_lower,
                            "post_date": date_str,
                            "reddit_id": submission.id,
                            "packaged": False,
                        })
                    else:
                        print(f"Download failed at {submission.url}")

                # Handle videos
                elif submission.is_video and submission.media and "reddit_video" in submission.media:
                    video_url = submission.media["reddit_video"]["fallback_url"]
                    frame_check = f"{base_name}_frame_0.jpg"

                    if frame_check in existing_files:
                        skipped_count += 1
                        continue

                    video_filepath = os.path.join(REDDIT_STAGING_DIR, f"{base_name}_temp.mp4")

                    if download_file(video_url, video_filepath):
                        frame_filenames = extract_video_frames(
                            video_filepath, REDDIT_STAGING_DIR, base_name, num_frames=NUM_OF_VIDEO_FRAMES
                        )

                        for frame_filename in frame_filenames:
                            new_count += 1
                            append_metadata({
                                "filename": frame_filename,
                                "label": label,
                                "subreddit": sub_lower,
                                "post_date": date_str,
                                "reddit_id": submission.id,
                                "packaged": False,
                            })

                        if os.path.exists(video_filepath):
                            os.remove(video_filepath)
                    else:
                        print(f"Video download failed at {video_url}")

            print(f"  r/{sub_name}: {new_count} new, {skipped_count} skipped (already scraped)")

        except Exception as e:
            print(f"Error scanning r/{sub_name}: {e}. Skipping.", flush=True)
            failed_subreddits.append((sub_name, str(e)))
            continue
    
    if failed_subreddits:
        print(f"\n{'='*60}", flush=True)
        print(f"WARNING: {len(failed_subreddits)} subreddits failed:", flush=True)
        for sub, err in failed_subreddits:
            print(f"  r/{sub}: {err}", flush=True)
        print(f"Consider removing these from the target list.", flush=True)
        print(f"{'='*60}", flush=True)

    if args_parsed.skip_nsfw_filter:
        print("Skipping NSFW image filter.")
        return

    print("Submitted NSFW image filter request")
    process = subprocess.run(
        [
            "sbatch",
            "--wait",
            f"--output={args_parsed.slurm_log_dir}/{TODAY}/reddit_filtering-%x_%j.out",
            args_parsed.nsfw_script,
            REDDIT_STAGING_DIR,
        ],
        capture_output=True,
        text=True,
    )

    print("job submitted. Filtering images...")
    print("see slurm logs dir to see status")

    if process.returncode == 0:
        print(f"Filtering successful.")
    else:
        print(f"ERROR: Filtering failed")
        print(f"Slurm Error Details: {process.stderr}")


if __name__ == "__main__":
    run_reddit_scraper()
