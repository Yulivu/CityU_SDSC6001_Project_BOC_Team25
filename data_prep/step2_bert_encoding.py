#!/usr/bin/env python3
"""
Step 2: Generate BERT embeddings for businesses using reviews
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import os
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_SEED = 42
SAVE_SHARD_EVERY = 2000
MAX_REVIEWS = 10
MAX_TOKENS = 128
MODEL_NAME = "distilbert-base-uncased"
TEXT_BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    """Load filtered reviews and training users"""
    print("Loading data...")
    
    # Load filtered reviews
    df_reviews = pd.read_parquet(
        OUTPUT_DIR / "reviews_filtered.parquet",
        columns=["user_id", "business_id", "text"],
    )
    
    # Load training users
    df_train_users = pd.read_parquet(OUTPUT_DIR / "users_train.parquet")
    train_users = set(df_train_users['user_id'])
    
    # Filter reviews to training set only
    df_reviews_train = df_reviews[df_reviews['user_id'].isin(train_users)]
    
    print(f"Loaded {len(df_reviews_train)} training reviews")
    return df_reviews_train

def initialize_bert():
    """Initialize BERT model and tokenizer"""
    print("Initializing BERT model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded on device: {DEVICE}")
    return tokenizer, model

def encode_texts(texts, tokenizer, model):
    if not texts:
        return None

    cleaned = [t for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned:
        return None

    all_embeddings = []
    for start in range(0, len(cleaned), TEXT_BATCH_SIZE):
        batch_texts = cleaned[start : start + TEXT_BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=MAX_TOKENS,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            if DEVICE.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
            all_embeddings.append(cls)

    emb = np.concatenate(all_embeddings, axis=0)
    return emb.mean(axis=0).astype(np.float32)

def shard_dir():
    return OUTPUT_DIR / "bert_embeddings_shards"

def list_processed_business_ids():
    sdir = shard_dir()
    if not sdir.exists():
        return set()
    processed = set()
    for f in sorted(sdir.glob("part_*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["business_id"])
            processed.update(df["business_id"].astype(str).tolist())
        except Exception:
            continue
    return processed

def next_shard_path():
    sdir = shard_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    existing = sorted(sdir.glob("part_*.parquet"))
    if not existing:
        idx = 0
    else:
        last = existing[-1].stem.split("_")[-1]
        idx = int(last) + 1 if last.isdigit() else len(existing)
    return sdir / f"part_{idx:05d}.parquet"

def generate_embeddings(df_reviews_train, tokenizer, model):
    print("Generating BERT embeddings...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed = list_processed_business_ids()

    df = df_reviews_train[["business_id", "text"]].copy()
    df["business_id"] = df["business_id"].astype(str)
    df = df.dropna(subset=["text"])
    df = df.sample(frac=1.0, random_state=RANDOM_SEED)
    df = df.groupby("business_id", sort=False).head(MAX_REVIEWS)
    business_texts = df.groupby("business_id", sort=False)["text"].apply(list).to_dict()

    all_business_ids = list(business_texts.keys())
    businesses_to_process = [bid for bid in all_business_ids if bid not in processed]

    print(f"Already processed: {len(processed)} businesses")
    print(f"To process: {len(businesses_to_process)} businesses")

    rows = []
    shard_count = 0
    for business_id in tqdm(businesses_to_process):
        emb = encode_texts(business_texts[business_id], tokenizer, model)
        if emb is None:
            continue
        row = {"business_id": business_id}
        for i, v in enumerate(emb.tolist()):
            row[f"dim_{i}"] = float(v)
        rows.append(row)

        if len(rows) >= SAVE_SHARD_EVERY:
            out_path = next_shard_path()
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            shard_count += 1
            rows = []

    if rows:
        out_path = next_shard_path()
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        shard_count += 1

    shard_files = sorted(shard_dir().glob("part_*.parquet"))
    if not shard_files:
        raise RuntimeError("No shard files were generated. Check your inputs and filters.")

    parts = [pd.read_parquet(f) for f in shard_files]
    final_df = pd.concat(parts, axis=0, ignore_index=True)
    final_df.to_parquet(OUTPUT_DIR / "business_bert_embeddings.parquet", index=False)

    print(f"Saved embeddings for {len(final_df)} businesses")
    print(f"Embedding dimension: {final_df.shape[1] - 1}")
    print(f"Shard files written: {shard_count}")

def main():
    """Main execution function"""
    print("Starting Step 2: BERT Encoding")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    df_reviews = load_data()
    
    # Step 2: Initialize BERT
    tokenizer, model = initialize_bert()
    
    # Step 3: Generate embeddings
    generate_embeddings(df_reviews, tokenizer, model)
    
    print("\nStep 2 completed successfully!")

if __name__ == "__main__":
    main()
