#!/usr/bin/env python3
"""
Step 4: Construct user-business and user-user graphs
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Yelp_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_SEED = 42

def load_training_data():
    """Load training users and reviews"""
    print("Loading training data...")
    
    # Load training users
    df_train_users = pd.read_parquet(OUTPUT_DIR / "users_train.parquet")
    train_users = set(df_train_users['user_id'])
    
    # Load filtered reviews
    df_reviews = pd.read_parquet(
        OUTPUT_DIR / "reviews_filtered.parquet",
        columns=["user_id", "business_id", "stars"],
    )
    
    # Filter to training users only
    df_reviews_train = df_reviews[df_reviews['user_id'].isin(train_users)]
    
    print(f"Loaded {len(train_users)} training users")
    print(f"Loaded {len(df_reviews_train)} training reviews")
    
    return train_users, df_reviews_train

def build_user_business_graph(df_reviews_train, train_users):
    """Build user-business bipartite graph"""
    print("Building user-business graph...")
    
    df_edges = df_reviews_train[["user_id", "business_id", "stars"]].copy()
    df_edges["weight"] = (df_edges["stars"].astype(float) / 5.0).astype(np.float32)
    df_edges = df_edges.drop(columns=["stars"])
    df_edges["user_id"] = df_edges["user_id"].astype(str)
    df_edges["business_id"] = df_edges["business_id"].astype(str)
    print(f"User-business graph: {len(df_edges)} edges")
    return df_edges

def parse_friends_field(friends_str):
    """Parse friends field from user data"""
    if pd.isna(friends_str) or not isinstance(friends_str, str):
        return []
    
    # Friends field is comma-separated list of user IDs
    friend_ids = [friend_id.strip() for friend_id in friends_str.split(',')]
    return [fid for fid in friend_ids if fid and fid != "None"]

def build_user_social_graph(train_users):
    """Build user-user social graph from friends data"""
    print("Building user-social graph...")
    
    user_file = DATA_DIR / "yelp_academic_dataset_user.json"
    if not user_file.exists():
        raise FileNotFoundError(f"User file not found: {user_file}")

    edges = set()
    with open(user_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            user_id = data.get("user_id")
            if not user_id or user_id not in train_users:
                continue

            friends = parse_friends_field(data.get("friends", ""))
            for friend_id in friends:
                if friend_id in train_users and friend_id != user_id:
                    a, b = (user_id, friend_id) if user_id < friend_id else (friend_id, user_id)
                    edges.add((a, b))

    df_unique_edges = pd.DataFrame(list(edges), columns=["user_id_1", "user_id_2"])
    print(f"User-social graph: {len(df_unique_edges)} unique edges")
    return df_unique_edges

def compute_graph_statistics(df_user_business_edges, df_user_social_edges, train_users, df_reviews_train):
    """Compute and print graph statistics"""
    print("Computing graph statistics...")
    
    # Get unique businesses from training set
    train_businesses = set(df_reviews_train['business_id'])
    
    # User-business graph stats
    ub_nodes = len(train_users) + len(train_businesses)
    ub_edges = len(df_user_business_edges)
    ub_avg_degree_user = ub_edges / len(train_users) if train_users else 0
    ub_avg_degree_business = ub_edges / len(train_businesses) if train_businesses else 0
    
    # User-social graph stats
    us_nodes = len(train_users)
    us_edges = len(df_user_social_edges)
    us_avg_degree = (2 * us_edges) / us_nodes if us_nodes else 0
    
    print("\nGraph Statistics:")
    print("=================")
    print("User-Business Bipartite Graph:")
    print(f"  User nodes: {len(train_users)}")
    print(f"  Business nodes: {len(train_businesses)}")
    print(f"  Total nodes: {ub_nodes}")
    print(f"  Edges: {ub_edges}")
    print(f"  Average user degree: {ub_avg_degree_user:.2f}")
    print(f"  Average business degree: {ub_avg_degree_business:.2f}")
    
    print("\nUser-Social Graph:")
    print(f"  User nodes: {us_nodes}")
    print(f"  Edges: {us_edges}")
    print(f"  Average degree: {us_avg_degree:.2f}")
    
    # Save statistics to file
    with open(OUTPUT_DIR / "graph_stats.txt", 'w') as f:
        f.write("Graph Construction Statistics\n")
        f.write("============================\n\n")
        
        f.write("User-Business Bipartite Graph:\n")
        f.write(f"  User nodes: {len(train_users)}\n")
        f.write(f"  Business nodes: {len(train_businesses)}\n")
        f.write(f"  Total nodes: {ub_nodes}\n")
        f.write(f"  Edges: {ub_edges}\n")
        f.write(f"  Average user degree: {ub_avg_degree_user:.2f}\n")
        f.write(f"  Average business degree: {ub_avg_degree_business:.2f}\n\n")
        
        f.write("User-Social Graph:\n")
        f.write(f"  User nodes: {us_nodes}\n")
        f.write(f"  Edges: {us_edges}\n")
        f.write(f"  Average degree: {us_avg_degree:.2f}\n")

def save_graphs(df_user_business_edges, df_user_social_edges):
    """Save both graphs"""
    print("Saving graphs...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save user-business edges
    df_user_business_edges.to_parquet(OUTPUT_DIR / "graph_user_business_edges.parquet", index=False)
    
    # Save user-social edges
    df_user_social_edges.to_parquet(OUTPUT_DIR / "graph_user_social_edges.parquet", index=False)
    
    print("Graphs saved successfully")

def main():
    """Main execution function"""
    print("Starting Step 4: Graph Construction")
    print("=" * 50)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load training data
    train_users, df_reviews_train = load_training_data()
    
    # Step 2: Build user-business graph
    df_user_business_edges = build_user_business_graph(df_reviews_train, train_users)
    
    # Step 3: Build user-social graph
    df_user_social_edges = build_user_social_graph(train_users)
    
    # Step 4: Compute statistics
    compute_graph_statistics(df_user_business_edges, df_user_social_edges, train_users, df_reviews_train)
    
    # Step 5: Save graphs
    save_graphs(df_user_business_edges, df_user_social_edges)
    
    print("\nStep 4 completed successfully!")

if __name__ == "__main__":
    main()
