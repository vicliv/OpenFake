import os
import shutil
import pandas as pd
import argparse

def sample_real_images(metadata_path, source_dir, output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(metadata_path)
    
    real_df = df[df["label"] == "real"]
    
    if len(real_df) < num_samples:
        num_samples = len(real_df)
        
    sampled_df = real_df.sample(n=num_samples)

    copied_count = 0
    for filename in sampled_df["filename"]:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(output_dir, filename)

        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            copied_count += 1
        else:
            print(f"File missing: {source_path}")

    print(f"Copied {copied_count} files to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/reddit_images")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output-dir", default="data/reddit_samples")
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()
    metadata = args.metadata or os.path.join(args.source_dir, "reddit_metadata.csv")
    sample_real_images(metadata, args.source_dir, args.output_dir, num_samples=args.num_samples)
