#!/usr/bin/env python3
"""
Exploratory Data Analysis for Yelp Dataset
Priority Analyses: Aspect Keyword Fallback Rate & Social Graph Sparsity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import json
import os
from pathlib import Path


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# City list (14 cities, Edmonton removed, 7 new cities added)
CITIES = ['Philadelphia', 'Tampa', 'Indianapolis', 'Tucson', 
          'Nashville', 'New Orleans', 'Reno', 'Saint Louis',
          'Santa Barbara', 'Boise', 'Clearwater', 'Metairie', 'Wilmington']

# Aspect keywords
ASPECT_KEYWORDS = {
    'food': ['food', 'taste', 'flavor', 'delicious', 'menu', 'dish'],
    'service': ['service', 'staff', 'waiter', 'waitress', 'friendly', 'rude'],
    'atmosphere': ['atmosphere', 'ambiance', 'decor', 'vibe', 'cozy', 'noisy', 'clean'],
    'price': ['price', 'cheap', 'expensive', 'value', 'worth', 'affordable']
}


def load_data(data_dir):
    """Load only necessary JSON datasets"""
    print("Loading datasets...")
    
    # Only load user data for social graph analysis
    print("  - Loading user data...")
    users = []
    with open(os.path.join(data_dir, 'yelp_academic_dataset_user.json'), 'r', encoding='utf-8') as f:
        for line in f:
            users.append(json.loads(line))
    users_df = pd.DataFrame(users)
    print(f"    Loaded {len(users_df)} users")
    
    # Load sample of review data for aspect analysis (600,000 rows)
    print("  - Loading review data sample (600,000 rows) for aspect analysis...")
    reviews = []
    with open(os.path.join(data_dir, 'yelp_academic_dataset_review.json'), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 600000:
                break
            reviews.append(json.loads(line))
    reviews_sample_df = pd.DataFrame(reviews)
    print(f"    Loaded {len(reviews_sample_df)} reviews for aspect analysis")
    
    # Load ALL reviews for cross-city user identification (social graph analysis)
    # Only load user_id and business_id fields to save memory
    print("  - Loading ALL reviews for cross-city user identification (user_id and business_id only)...")
    all_reviews = []
    with open(os.path.join(data_dir, 'yelp_academic_dataset_review.json'), 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            all_reviews.append({
                'user_id': review['user_id'],
                'business_id': review['business_id']
            })
    all_reviews_df = pd.DataFrame(all_reviews)
    print(f"    Loaded {len(all_reviews_df):,} total reviews for cross-city analysis")
    
    # Load business data to get city information for cross-city user identification
    print("  - Loading business data...")
    business = []
    with open(os.path.join(data_dir, 'yelp_academic_dataset_business.json'), 'r', encoding='utf-8') as f:
        for line in f:
            business.append(json.loads(line))
    business_df = pd.DataFrame(business)
    print(f"    Loaded {len(business_df)} businesses")
    
    # Standardize city names: merge "St. Louis" into "Saint Louis"
    print("  - Standardizing city names...")
    st_louis_count = (business_df['city'] == 'St. Louis').sum()
    business_df['city'] = business_df['city'].replace('St. Louis', 'Saint Louis')
    print(f"    Merged {st_louis_count} businesses from 'St. Louis' into 'Saint Louis'")
    
    return reviews_sample_df, all_reviews_df, users_df, business_df








def analyze_aspect_keyword_fallback(reviews_df, output_dir):
    """1. Aspect keyword fallback rate"""
    print("\n" + "="*60)
    print("1. ASPECT KEYWORD FALLBACK RATE")
    print("="*60)
    
    # Use all loaded reviews (already limited to 200,000 in load_data)
    sampled_reviews = reviews_df.copy()
    sample_size = len(sampled_reviews)
    print(f"  Processing {sample_size:,} reviews...")
    
    # Assign aspect based on keyword matching (priority order)
    print("  Assigning aspects using keyword matching...")
    
    def assign_aspect(text):
        text_lower = text.lower()
        for aspect, keywords in ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return aspect
        return 'no_match'
    
    tqdm.pandas(desc="Matching keywords")
    sampled_reviews['aspect'] = sampled_reviews['text'].progress_apply(assign_aspect)
    
    # Calculate distribution
    aspect_counts = sampled_reviews['aspect'].value_counts()
    aspect_percentages = (aspect_counts / len(sampled_reviews) * 100).round(2)
    
    print("\n  Aspect distribution:")
    for aspect, pct in aspect_percentages.items():
        print(f"    {aspect}: {pct}% ({aspect_counts[aspect]:.0f} reviews)")
    
    # Plot pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c7c7c7']
    ax.pie(aspect_percentages.values, labels=aspect_percentages.index, autopct='%1.1f%%', 
           colors=colors, startangle=90)
    ax.set_title('Aspect Distribution in Reviews (Keyword Matching)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aspect_keyword_fallback.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved figure: aspect_keyword_fallback.png")
    
    return aspect_percentages.to_dict()





def analyze_social_graph_sparsity(reviews_df, users_df, business_df, output_dir):
    """2. Social graph sparsity - for cross-city users only"""
    print("\n" + "="*60)
    print("2. SOCIAL GRAPH SPARSITY (CROSS-CITY USERS)")
    print("="*60)
    
    # Filter businesses to target cities
    business_df = business_df[business_df['city'].isin(CITIES)].copy()
    print(f"  Filtered to {len(business_df):,} businesses in target cities")
    
    # Merge reviews with business data to get city information
    print("  Merging reviews with business data...")
    reviews_df = reviews_df.merge(
        business_df[['business_id', 'city']], 
        on='business_id', 
        how='inner'
    )
    print(f"  Merged to {len(reviews_df):,} reviews in target cities")
    
    # Parse friends field correctly (handle comma-separated strings)
    def parse_friends(friends_field):
        """Parse friends field - handle comma-separated strings"""
        if friends_field is None:
            return []
        if isinstance(friends_field, list):
            return friends_field
        if isinstance(friends_field, str):
            # Handle comma-separated string
            if friends_field.strip() == '' or friends_field.strip().lower() == 'none':
                return []
            # Split by comma and clean up
            return [f.strip() for f in friends_field.split(',') if f.strip()]
        return []
    
    print("  Parsing friends field...")
    users_df['friends_parsed'] = users_df['friends'].apply(parse_friends)
    users_df['friend_count'] = users_df['friends_parsed'].apply(len)
    
    # Find cross-city users (active in >=2 cities)
    print("  Identifying cross-city users...")
    user_city_counts = reviews_df.groupby('user_id')['city'].nunique()
    cross_city_users = user_city_counts[user_city_counts >= 2].index
    print(f"  Found {len(cross_city_users):,} cross-city users")
    
    # Filter to cross-city users only
    cross_city_users_df = users_df[users_df['user_id'].isin(cross_city_users)].copy()
    print(f"  Matched {len(cross_city_users_df):,} users with friend data")
    
    # Statistics for cross-city users
    users_with_friends = (cross_city_users_df['friend_count'] > 0).sum()
    pct_with_friends = (users_with_friends / len(cross_city_users_df) * 100)
    
    print(f"\n  Cross-city users statistics:")
    print(f"  Total cross-city users: {len(cross_city_users_df):,}")
    print(f"  Users with at least 1 friend: {users_with_friends:,}")
    print(f"  % with friends: {pct_with_friends:.2f}%")
    print(f"  Mean friend count: {cross_city_users_df['friend_count'].mean():.2f}")
    print(f"  Median friend count: {cross_city_users_df['friend_count'].median():.2f}")
    
    # Bucket distribution
    print(f"\n  Friend count distribution:")
    bucket_0 = (cross_city_users_df['friend_count'] == 0).sum()
    bucket_1_5 = ((cross_city_users_df['friend_count'] >= 1) & (cross_city_users_df['friend_count'] <= 5)).sum()
    bucket_6_10 = ((cross_city_users_df['friend_count'] >= 6) & (cross_city_users_df['friend_count'] <= 10)).sum()
    bucket_11_50 = ((cross_city_users_df['friend_count'] >= 11) & (cross_city_users_df['friend_count'] <= 50)).sum()
    bucket_51_100 = ((cross_city_users_df['friend_count'] >= 51) & (cross_city_users_df['friend_count'] <= 100)).sum()
    bucket_100plus = (cross_city_users_df['friend_count'] > 100).sum()
    
    print(f"    0 friends: {bucket_0:,} ({bucket_0/len(cross_city_users_df)*100:.2f}%)")
    print(f"    1-5 friends: {bucket_1_5:,} ({bucket_1_5/len(cross_city_users_df)*100:.2f}%)")
    print(f"    6-10 friends: {bucket_6_10:,} ({bucket_6_10/len(cross_city_users_df)*100:.2f}%)")
    print(f"    11-50 friends: {bucket_11_50:,} ({bucket_11_50/len(cross_city_users_df)*100:.2f}%)")
    print(f"    51-100 friends: {bucket_51_100:,} ({bucket_51_100/len(cross_city_users_df)*100:.2f}%)")
    print(f"    100+ friends: {bucket_100plus:,} ({bucket_100plus/len(cross_city_users_df)*100:.2f}%)")
    
    # Plot distribution (clipped at 200)
    friend_counts_clipped = cross_city_users_df['friend_count'].clip(upper=200)
    
    plt.figure(figsize=(10, 6))
    plt.hist(friend_counts_clipped, bins=50, alpha=0.7, color='coral', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='0 friends')
    plt.xlabel('Friend Count (clipped at 200)')
    plt.ylabel('Number of Users')
    plt.title(f'Distribution of Friend Counts for Cross-City Users\n(n={len(cross_city_users_df):,})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'social_graph_sparsity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved figure: social_graph_sparsity.png")
    
    return {
        'total_cross_city_users': len(cross_city_users_df),
        'users_with_friends': int(users_with_friends),
        'pct_with_friends': float(pct_with_friends),
        'mean_friends': float(cross_city_users_df['friend_count'].mean()),
        'median_friends': float(cross_city_users_df['friend_count'].median()),
        'bucket_0': int(bucket_0),
        'bucket_1_5': int(bucket_1_5),
        'bucket_6_10': int(bucket_6_10),
        'bucket_11_50': int(bucket_11_50),
        'bucket_51_100': int(bucket_51_100),
        'bucket_100plus': int(bucket_100plus)
    }


def generate_report(results, output_path):
    """Generate summary report"""
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("Yelp Dataset Analysis (Priority Analyses Only)\n")
        f.write("="*60 + "\n\n")
        
        # 1. Aspect keyword fallback
        f.write("1. ASPECT KEYWORD FALLBACK RATE\n")
        f.write("-"*60 + "\n")
        f.write("Aspect distribution (keyword matching):\n")
        for aspect, pct in results['aspect_fallback'].items():
            f.write(f"  {aspect}: {pct}%\n")
        f.write("\n")
        
        # 2. Social graph sparsity
        f.write("2. SOCIAL GRAPH SPARSITY (CROSS-CITY USERS)\n")
        f.write("-"*60 + "\n")
        f.write(f"  Total cross-city users: {results['social']['total_cross_city_users']:,}\n")
        f.write(f"  Users with at least 1 friend: {results['social']['users_with_friends']:,}\n")
        f.write(f"  % with friends: {results['social']['pct_with_friends']:.2f}%\n")
        f.write(f"  Mean friend count: {results['social']['mean_friends']:.2f}\n")
        f.write(f"  Median friend count: {results['social']['median_friends']:.2f}\n")
        f.write(f"\n  Friend count distribution:\n")
        f.write(f"    0 friends: {results['social']['bucket_0']:,}\n")
        f.write(f"    1-5 friends: {results['social']['bucket_1_5']:,}\n")
        f.write(f"    6-10 friends: {results['social']['bucket_6_10']:,}\n")
        f.write(f"    11-50 friends: {results['social']['bucket_11_50']:,}\n")
        f.write(f"    51-100 friends: {results['social']['bucket_51_100']:,}\n")
        f.write(f"    100+ friends: {results['social']['bucket_100plus']:,}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*60 + "\n")
    
    print(f"  Report saved to: {output_path}")


def main():
    """Main execution"""
    # Get data directory (Yelp_data is sibling folder to eda)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'Yelp_data')
    
    # Create output directories
    output_dir = os.path.join(project_root, 'eda_output')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS FOR YELP DATASET")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data (optimized - only what we need)
    reviews_sample_df, all_reviews_df, users_df, business_df = load_data(data_dir)
    
    # Run priority analyses only
    results = {}
    
    # 1. Aspect keyword fallback rate (fast - uses sampled reviews)
    results['aspect_fallback'] = analyze_aspect_keyword_fallback(reviews_sample_df, figures_dir)
    
    # 2. Social graph sparsity (for cross-city users only - uses all reviews)
    results['social'] = analyze_social_graph_sparsity(all_reviews_df, users_df, business_df, figures_dir)
    
    # Generate report
    report_path = os.path.join(output_dir, 'eda_report.txt')
    generate_report(results, report_path)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Figures saved to: {figures_dir}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
