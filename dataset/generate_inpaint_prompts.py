"""
Generates inpainting replacement prompts using Gemma for Open Images segmentation masks.
Each SLURM array task processes a shard of the data.

Context is built from multiple sources (best available used):
  1. Localized Narratives (full image captions) — best quality
  2. Visual Relationships (e.g. "Man wearing Tie") — good quality
  3. Scene objects (all segmented classes in the image) — decent quality
  4. Class name only — fallback

Usage:
    # Generate shard (called via sbatch array):
    python -u generate_prompts.py --shard-index $SLURM_ARRAY_TASK_ID

    # Merge all shards into final CSV (run after all shards complete):
    python -u generate_prompts.py --merge-only
"""

import argparse
import json
import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

METADATA_DIR = "data/open_images/zoo/open-images-v7/train/metadata"
LABELS_DIR = "data/open_images/zoo/open-images-v7/train/labels"
OUTPUT_DIR = "data/open_images/prompt_shards"
FINAL_OUTPUT = "data/open_images/master_prompts_300k.csv"
MODEL_ID = "google/gemma-3-27b-it"

NARRATIVES_FILE = os.path.join(LABELS_DIR, "localized_narratives_train.jsonl")
RELATIONSHIPS_FILE = os.path.join(LABELS_DIR, "oidv6-train-annotations-vrd.csv")

BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-count", type=int, default=300000)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--metadata-dir", default=METADATA_DIR)
    parser.add_argument("--labels-dir", default=LABELS_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--final-output", default=FINAL_OUTPUT)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_class_names():
    """Load the MID -> human-readable class name mapping."""
    class_file = os.path.join(METADATA_DIR, "classes.csv")
    class_df = pd.read_csv(class_file, header=None, names=["LabelName", "ClassName"])
    return dict(zip(class_df["LabelName"], class_df["ClassName"]))


def load_narratives():
    """Load localized narratives into a dict: ImageID -> caption string."""
    if not os.path.exists(NARRATIVES_FILE):
        print(f"Narratives file not found at {NARRATIVES_FILE}, skipping.", flush=True)
        return {}

    print(f"Loading localized narratives...", flush=True)
    narratives = {}
    with open(NARRATIVES_FILE, "r") as f:
        for line in f:
            record = json.loads(line)
            image_id = record.get("image_id", "")
            caption = record.get("caption", "")
            if image_id and caption:
                # Keep only the first narrative per image (some have multiple)
                if image_id not in narratives:
                    # Truncate very long captions to keep prompt manageable
                    narratives[image_id] = caption[:300]

    print(f"Loaded {len(narratives)} narratives.", flush=True)
    return narratives


def load_relationships(class_names):
    """
    Load visual relationships into a dict: ImageID -> list of relationship strings.
    e.g. {"abc123": ["Man wearing Tie", "Cup on Table"]}
    """
    if not os.path.exists(RELATIONSHIPS_FILE):
        print(f"Relationships file not found at {RELATIONSHIPS_FILE}, skipping.", flush=True)
        return {}

    print(f"Loading visual relationships...", flush=True)
    vrd_df = pd.read_csv(RELATIONSHIPS_FILE)
    print(f"Loaded {len(vrd_df)} relationship annotations.", flush=True)

    # Vectorized: map MIDs to class names
    vrd_df["Name1"] = vrd_df["LabelName1"].map(class_names).fillna("Object")
    vrd_df["Name2"] = vrd_df["LabelName2"].map(class_names).fillna("Object")
    # Column might be RelationshipLabel or RelationLabel depending on version
    rel_col = "RelationshipLabel" if "RelationshipLabel" in vrd_df.columns else "RelationLabel"
    vrd_df["Triplet"] = vrd_df["Name1"] + " " + vrd_df[rel_col] + " " + vrd_df["Name2"]

    # Group by ImageID, cap at 8 per image
    relationships = (
        vrd_df.groupby("ImageID")["Triplet"]
        .apply(lambda x: x.head(8).tolist())
        .to_dict()
    )

    print(f"Built relationships for {len(relationships)} images.", flush=True)
    return relationships


def build_scene_objects(seg_df, class_names):
    """
    Build a dict: ImageID -> list of unique class names in the image.
    e.g. {"abc123": ["Person", "Tie", "Chair"]}
    """
    print(f"Building scene object lists...", flush=True)
    # Map LabelName to ClassName
    seg_df = seg_df.copy()
    seg_df["_ClassName"] = seg_df["LabelName"].map(class_names)
    seg_df = seg_df.dropna(subset=["_ClassName"])

    scene = (
        seg_df.groupby("ImageID")["_ClassName"]
        .apply(lambda x: sorted(x.unique().tolist()))
        .to_dict()
    )

    print(f"Built scene object lists for {len(scene)} images.", flush=True)
    return scene


def prepare_mask_metadata(total_count):
    print(f"Loading segmentation and class data...", flush=True)
    seg_file = os.path.join(LABELS_DIR, "segmentations.csv")
    seg_df = pd.read_csv(seg_file)
    print(f"Loaded {len(seg_df)} segmentation rows.", flush=True)

    class_names = load_class_names()

    # Merge class names onto segmentation data
    class_df = pd.DataFrame(list(class_names.items()), columns=["LabelName", "ClassName"])
    merged_df = pd.merge(seg_df, class_df, on="LabelName", how="left")

    target_df = merged_df[["ImageID", "MaskPath", "LabelName", "ClassName"]].head(total_count).copy()

    print(f"Prepared {len(target_df)} rows for prompt generation.", flush=True)
    return target_df, seg_df, class_names


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------
def get_context_for_row(row, narratives, relationships, scene_objects):
    """
    Build the best available context string for a given row.
    Returns (context_string, tier_number).
    """
    image_id = row["ImageID"]
    target_object = row["ClassName"]

    # Tier 1: Localized narrative (full caption)
    if image_id in narratives:
        caption = narratives[image_id]
        return f'Image description: "{caption}"\nObject to replace: {target_object}', 1

    # Tier 2: Visual relationships
    if image_id in relationships:
        rels = relationships[image_id]
        rel_str = "; ".join(rels)
        return f"Scene relationships: {rel_str}\nObject to replace: {target_object}", 2

    # Tier 3: Scene objects (all objects in the image)
    if image_id in scene_objects:
        objects = scene_objects[image_id]
        obj_str = ", ".join(objects)
        return f"Objects in the image: {obj_str}\nObject to replace: {target_object}", 3

    # Tier 4: Just the class name
    return f"Object to replace: {target_object}", 4


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer():
    print(f"Loading tokenizer from {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {MODEL_ID}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    print(f"Model loaded successfully.", flush=True)

    return tokenizer, model


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------
def build_user_message(row, narratives, relationships, scene_objects):
    """Build a prompt asking for a JSON-structured inpainting prompt."""
    image_id = row["ImageID"]
    target = row["ClassName"]

    # Build context from best available source
    if image_id in narratives:
        context = f'This photo is described as: "{narratives[image_id]}"'
    elif image_id in relationships:
        rels = "; ".join(relationships[image_id])
        context = f"This photo shows: {rels}"
    elif image_id in scene_objects:
        objs = ", ".join(scene_objects[image_id])
        context = f"This photo contains: {objs}"
    else:
        context = f"A photo contains a {target}"

    return (
        f"{context}\n\n"
        f"I'm going to use an AI inpainting model on the {target} in this image. "
        f"Give me a prompt describing what the {target} area should be changed to after inpainting. "
        f"Think about things like, but not limited to: changing the material or texture, applying a different pattern or color scheme, "
        f"swapping it for a similar but different object, changing the style or era, removing it to reveal what's behind, "
        f"or making it look like it's made of an unexpected substance. Keep it grounded and realistic — "
        f"avoid fantasy, sci-fi, glowing, bioluminescent, or ethereal themes. "
        f'Respond with a JSON object containing a single key "prompt" with your inpainting prompt as the value.'
    )


def generate_prompts_batch(tokenizer, model, batch_df, device,
                           narratives, relationships, scene_objects):
    """Generate inpainting prompts in a batch."""
    messages_batch = []
    for _, row in batch_df.iterrows():
        user_message = build_user_message(row, narratives, relationships, scene_objects)
        messages_batch.append([{"role": "user", "content": user_message}])

    formatted_prompts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_batch
    ]

    # Track per-sample input lengths before padding
    input_lengths = []
    for prompt in formatted_prompts:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_lengths.append(ids.shape[1])

    inputs = tokenizer(
        formatted_prompts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.9,
        )

    results = []
    for j in range(len(batch_df)):
        response = tokenizer.decode(
            outputs[j][input_lengths[j]:], skip_special_tokens=True
        ).strip()
        results.append(extract_prompt(response))

    return results


def extract_prompt(response):
    """Extract the prompt value from JSON response. No cleaning, just parse."""
    import json as json_module

    # Try direct JSON parse
    try:
        data = json_module.loads(response)
        if isinstance(data, dict) and "prompt" in data:
            return data["prompt"].strip()
    except (json_module.JSONDecodeError, ValueError):
        pass

    # Try to find JSON embedded in the response (model sometimes adds text around it)
    json_match = re.search(r'\{[^{}]*"prompt"\s*:\s*"([^"]+)"[^{}]*\}', response)
    if json_match:
        return json_match.group(1).strip()

    # If no JSON at all, return the raw response as-is
    return response.strip()


# ---------------------------------------------------------------------------
# Shard generation and merging
# ---------------------------------------------------------------------------
def generate_shard(args):
    shard_index = args.shard_index
    total_count = args.total_count
    num_shards = args.num_shards

    print(f"=== Shard {shard_index}/{num_shards} starting. PID={os.getpid()} ===", flush=True)

    # Shard 0 cleans up old shards from previous runs
    if shard_index == 0 and os.path.exists(OUTPUT_DIR):
        import glob
        old_shards = glob.glob(os.path.join(OUTPUT_DIR, "shard_*.csv"))
        if old_shards:
            print(f"Cleaning up {len(old_shards)} old shard files...", flush=True)
            for f in old_shards:
                os.remove(f)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if this shard is already done
    shard_output = os.path.join(OUTPUT_DIR, f"shard_{shard_index}.csv")
    if os.path.exists(shard_output):
        existing = pd.read_csv(shard_output)
        print(f"Shard {shard_index}: Output file already exists with {len(existing)} rows. Skipping.", flush=True)
        return

    # Load all data sources
    target_df, full_seg_df, class_names = prepare_mask_metadata(total_count)
    narratives = load_narratives()
    relationships = load_relationships(class_names)
    scene_objects = build_scene_objects(full_seg_df, class_names)

    # Slice to this shard
    shard_size = total_count // num_shards
    start = shard_index * shard_size
    end = total_count if shard_index == num_shards - 1 else start + shard_size

    shard_df = target_df.iloc[start:end].reset_index(drop=True)
    print(f"Shard {shard_index}: Processing rows {start}-{end} ({len(shard_df)} rows).", flush=True)

    if len(shard_df) == 0:
        print(f"Shard {shard_index}: Empty shard, nothing to do.", flush=True)
        return

    # Count context tiers for logging
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for _, row in shard_df.iterrows():
        _, tier = get_context_for_row(row, narratives, relationships, scene_objects)
        tier_counts[tier] += 1
    print(f"Shard {shard_index}: Context tiers — "
          f"Narratives: {tier_counts[1]}, Relationships: {tier_counts[2]}, "
          f"Scene objects: {tier_counts[3]}, Class only: {tier_counts[4]}", flush=True)

    # Load model
    tokenizer, model = load_model_and_tokenizer()
    device = next(model.parameters()).device
    print(f"Shard {shard_index}: Model on device {device}.", flush=True)

    # Generate prompts in batches
    prompts = []
    for i in range(0, len(shard_df), BATCH_SIZE):
        batch_df = shard_df.iloc[i:i + BATCH_SIZE]
        batch_prompts = generate_prompts_batch(
            tokenizer, model, batch_df, device,
            narratives, relationships, scene_objects
        )
        prompts.extend(batch_prompts)

        processed = i + len(batch_df)
        if processed % 1024 < BATCH_SIZE or processed == len(shard_df):
            print(f"Shard {shard_index}: Processed {processed}/{len(shard_df)} prompts.", flush=True)

    # Save shard
    shard_df["generated_prompt"] = prompts
    shard_df.to_csv(shard_output, index=False)
    print(f"Shard {shard_index}: Saved {len(shard_df)} rows to {shard_output}", flush=True)


def merge_shards(args):
    """Merge all completed shard CSVs into the final master CSV."""
    print(f"Merging shards from {OUTPUT_DIR}...", flush=True)

    shard_files = sorted(
        [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
         if f.startswith("shard_") and f.endswith(".csv")]
    )

    if not shard_files:
        print(f"ERROR: No shard files found in {OUTPUT_DIR}.", flush=True)
        return

    print(f"Found {len(shard_files)} shard files: {[os.path.basename(f) for f in shard_files]}", flush=True)

    dfs = []
    for sf in shard_files:
        df = pd.read_csv(sf)
        print(f"  {os.path.basename(sf)}: {len(df)} rows", flush=True)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(FINAL_OUTPUT, index=False)
    print(f"Merged {len(merged)} total rows into {FINAL_OUTPUT}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    METADATA_DIR = args.metadata_dir
    LABELS_DIR = args.labels_dir
    OUTPUT_DIR = args.output_dir
    FINAL_OUTPUT = args.final_output
    NARRATIVES_FILE = os.path.join(LABELS_DIR, "localized_narratives_train.jsonl")
    RELATIONSHIPS_FILE = os.path.join(LABELS_DIR, "oidv6-train-annotations-vrd.csv")

    if args.merge_only:
        merge_shards(args)
    else:
        generate_shard(args)
