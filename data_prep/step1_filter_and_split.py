#!/usr/bin/env python3
"""
Step 1: Filter and split data for cross-city restaurant recommendation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Yelp_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_SEED = 42

# Target cities (merge "St. Louis" into "Saint Louis")
TARGET_CITIES = [
    "Philadelphia", "Tampa", "Indianapolis", "Tucson", "Nashville", 
    "New Orleans", "Reno", "Saint Louis", "Santa Barbara", "Boise", 
    "Clearwater", "Metairie", "Wilmington"
]

def load_and_filter_businesses():
    """Load and filter businesses to restaurants in target cities"""
    print("Loading business data...")
    
    # Read business data
    business_file = DATA_DIR / "yelp_academic_dataset_business.json"
    businesses = []
    
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                businesses.append(data)
            except json.JSONDecodeError:
                continue
    
    df_business = pd.DataFrame(businesses)
    
    # Standardize city names
    df_business['city'] = df_business['city'].str.strip()
    df_business['city'] = df_business['city'].replace({"St. Louis": "Saint Louis"})
    
    # Filter to target cities
    df_business = df_business[df_business['city'].isin(TARGET_CITIES)]
    
    # Filter to restaurants
    df_business = df_business[df_business['categories'].str.contains('Restaurants', na=False)]
    
    print(f"Found {len(df_business)} restaurants in target cities")
    return df_business

def load_and_filter_reviews(business_ids, df_business):
    """Load reviews for filtered businesses"""
    print("Loading review data...")
    
    review_file = DATA_DIR / "yelp_academic_dataset_review.json"
    reviews = []
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data['business_id'] in business_ids:
                    reviews.append(data)
            except json.JSONDecodeError:
                continue
    
    df_reviews = pd.DataFrame(reviews)
    
    # Add city information by merging with business data (using the DataFrame we already have)
    df_reviews = df_reviews.merge(df_business[['business_id', 'city']], on='business_id')
    
    print(f"Found {len(df_reviews)} reviews for filtered restaurants")
    return df_reviews

def identify_cross_city_users(df_reviews):
    """Identify users active in multiple cities"""
    print("Identifying cross-city users...")
    
    # Count cities per user
    user_city_counts = df_reviews.groupby('user_id')['city'].nunique()
    cross_city_users = user_city_counts[user_city_counts >= 2].index.tolist()
    
    print(f"Found {len(cross_city_users)} users active in multiple cities")
    return cross_city_users

def apply_user_filters(df_reviews, cross_city_users):
    """Apply source/target city review count filters"""
    print("Applying user filters...")
    
    # Filter to cross-city users
    df_filtered = df_reviews[df_reviews['user_id'].isin(cross_city_users)]
    
    # Calculate review counts per user per city
    user_city_counts = df_filtered.groupby(['user_id', 'city']).size().reset_index(name='city_review_count')
    
    # Identify users with at least one city having >= 5 reviews (source) 
    # and at least one other city having >= 1 review (target)
    valid_users = []
    
    for user_id in cross_city_users:
        user_cities = user_city_counts[user_city_counts['user_id'] == user_id]
        
        # Check if user has at least one city with >= 5 reviews
        has_source_city = any(user_cities['city_review_count'] >= 5)
        
        # Check if user has at least one other city with >= 1 review
        has_target_city = len(user_cities) >= 2  # At least 2 different cities
        
        if has_source_city and has_target_city:
            valid_users.append(user_id)
    
    df_filtered = df_filtered[df_filtered['user_id'].isin(valid_users)]
    
    print(f"After filtering: {len(df_filtered)} reviews from {len(valid_users)} users")
    return df_filtered

def split_users(df_reviews):
    """Split users into train/val/test sets"""
    print("Splitting users...")
    
    np.random.seed(RANDOM_SEED)
    
    # Get unique users
    unique_users = df_reviews['user_id'].unique()
    np.random.shuffle(unique_users)
    
    # Calculate split sizes
    n_users = len(unique_users)
    n_train = int(0.7 * n_users)
    n_val = int(0.1 * n_users)
    
    # Split users
    train_users = unique_users[:n_train]
    val_users = unique_users[n_train:n_train + n_val]
    test_users = unique_users[n_train + n_val:]
    
    print(f"Split: {len(train_users)} train, {len(val_users)} val, {len(test_users)} test users")
    
    return train_users, val_users, test_users

def save_results(df_business, df_reviews, train_users, val_users, test_users):
    """Save all outputs"""
    print("Saving results...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save filtered business data
    df_business.to_parquet(OUTPUT_DIR / "business_filtered.parquet", index=False)
    
    # Save filtered reviews
    df_reviews.to_parquet(OUTPUT_DIR / "reviews_filtered.parquet", index=False)
    
    # Save user splits
    pd.DataFrame({'user_id': train_users}).to_parquet(OUTPUT_DIR / "users_train.parquet", index=False)
    pd.DataFrame({'user_id': val_users}).to_parquet(OUTPUT_DIR / "users_val.parquet", index=False)
    pd.DataFrame({'user_id': test_users}).to_parquet(OUTPUT_DIR / "users_test.parquet", index=False)
    
    # Save split statistics
    train_reviews = len(df_reviews[df_reviews['user_id'].isin(train_users)])
    val_reviews = len(df_reviews[df_reviews['user_id'].isin(val_users)])
    test_reviews = len(df_reviews[df_reviews['user_id'].isin(test_users)])
    
    with open(OUTPUT_DIR / "split_stats.txt", 'w') as f:
        f.write("Split Statistics\n")
        f.write("===============\n")
        f.write(f"Total businesses: {len(df_business)}\n")
        f.write(f"Total reviews: {len(df_reviews)}\n")
        f.write(f"Total users: {len(train_users) + len(val_users) + len(test_users)}\n\n")
        f.write("Train Split:\n")
        f.write(f"  Users: {len(train_users)}\n")
        f.write(f"  Reviews: {train_reviews}\n\n")
        f.write("Validation Split:\n")
        f.write(f"  Users: {len(val_users)}\n")
        f.write(f"  Reviews: {val_reviews}\n\n")
        f.write("Test Split:\n")
        f.write(f"  Users: {len(test_users)}\n")
        f.write(f"  Reviews: {test_reviews}\n")

def main():
    """Main execution function"""
    print("Starting Step 1: Filter and Split Data")
    print("=" * 50)
    
    # Step 1: Filter businesses
    df_business = load_and_filter_businesses()
    
    # Step 2: Filter reviews for these businesses
    df_reviews = load_and_filter_reviews(set(df_business['business_id']), df_business)
    
    # Step 3: Identify cross-city users
    cross_city_users = identify_cross_city_users(df_reviews)
    
    # Step 4: Apply user filters
    df_reviews_filtered = apply_user_filters(df_reviews, cross_city_users)
    
    # Step 5: Split users
    train_users, val_users, test_users = split_users(df_reviews_filtered)
    
    # Step 6: Save results
    save_results(df_business, df_reviews_filtered, train_users, val_users, test_users)
    
    print("\nStep 1 completed successfully!")
    print("Output files saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
