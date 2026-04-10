#!/usr/bin/env python3
"""
Test script to fix friends field parsing in user.json
"""

import json
import os
import pandas as pd
from tqdm import tqdm


def load_user_sample(data_dir, n_samples=5):
    """Load sample users to inspect friends field format"""
    print("="*60)
    print("LOADING SAMPLE USERS TO INSPECT FRIENDS FIELD")
    print("="*60)
    
    users = []
    with open(os.path.join(data_dir, 'yelp_academic_dataset_user.json'), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            users.append(json.loads(line))
    
    return users


def inspect_friends_field(users):
    """Inspect raw friends field values"""
    print("\n" + "="*60)
    print("RAW FRIENDS FIELD VALUES (First 5 samples)")
    print("="*60)
    
    for i, user in enumerate(users):
        user_id = user.get('user_id', 'N/A')
        friends = user.get('friends', 'N/A')
        print(f"\nSample {i+1}:")
        print(f"  User ID: {user_id}")
        print(f"  Friends field type: {type(friends)}")
        print(f"  Friends field raw value: {repr(friends)}")
        print(f"  Friends field (first 200 chars): {str(friends)[:200]}")


def analyze_friends_field(data_dir):
    """Analyze the friends field in all users"""
    print("\n" + "="*60)
    print("ANALYZING FRIENDS FIELD IN ALL USERS")
    print("="*60)
    
    # Load all users
    users = []
    with open(os.path.join(data_dir, 'yelp_academic_dataset_user.json'), 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading users"):
            users.append(json.loads(line))
    
    users_df = pd.DataFrame(users)
    print(f"\nLoaded {len(users_df):,} users")
    
    # Inspect friends field statistics
    print("\nFriends field statistics:")
    print(f"  Type of friends column: {type(users_df['friends'].iloc[0])}")
    print(f"  Number of null values: {users_df['friends'].isnull().sum()}")
    
    # Count different types
    print("\nDistribution of friends field types:")
    type_counts = users_df['friends'].apply(type).value_counts()
    for dtype, count in type_counts.items():
        print(f"  {dtype}: {count:,} ({count/len(users_df)*100:.2f}%)")
    
    # Show sample values
    print("\nSample raw friends values (first 10):")
    for i in range(min(10, len(users_df))):
        friends = users_df['friends'].iloc[i]
        print(f"  {i}: {repr(str(friends)[:100])}")
    
    # Parse friends correctly
    print("\n" + "="*60)
    print("PARSING FRIENDS FIELD (CORRECTED)")
    print("="*60)
    
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
    
    users_df['friends_parsed'] = users_df['friends'].apply(parse_friends)
    users_df['friend_count'] = users_df['friends_parsed'].apply(len)
    
    # Statistics
    users_with_friends = (users_df['friend_count'] > 0).sum()
    pct_with_friends = (users_with_friends / len(users_df) * 100)
    
    print(f"\nResults after correct parsing:")
    print(f"  Total users: {len(users_df):,}")
    print(f"  Users with friends: {users_with_friends:,}")
    print(f"  % with friends: {pct_with_friends:.2f}%")
    print(f"  Mean friend count: {users_df['friend_count'].mean():.2f}")
    print(f"  Median friend count: {users_df['friend_count'].median():.2f}")
    print(f"  Max friend count: {users_df['friend_count'].max()}")
    
    # Distribution of friend counts
    print("\nDistribution of friend counts:")
    print(f"  0 friends: {(users_df['friend_count'] == 0).sum():,}")
    print(f"  1-5 friends: {((users_df['friend_count'] >= 1) & (users_df['friend_count'] <= 5)).sum():,}")
    print(f"  6-10 friends: {((users_df['friend_count'] >= 6) & (users_df['friend_count'] <= 10)).sum():,}")
    print(f"  11-50 friends: {((users_df['friend_count'] >= 11) & (users_df['friend_count'] <= 50)).sum():,}")
    print(f"  51-100 friends: {((users_df['friend_count'] >= 51) & (users_df['friend_count'] <= 100)).sum():,}")
    print(f"  100+ friends: {(users_df['friend_count'] > 100).sum():,}")
    
    # Show examples of users with friends
    print("\n" + "="*60)
    print("EXAMPLES OF USERS WITH FRIENDS")
    print("="*60)
    users_with_friends_df = users_df[users_df['friend_count'] > 0].head(5)
    for i, (_, user) in enumerate(users_with_friends_df.iterrows()):
        print(f"\nExample {i+1}:")
        print(f"  User ID: {user['user_id']}")
        print(f"  Friend count: {user['friend_count']}")
        print(f"  Friends (first 5): {user['friends_parsed'][:5]}")
        print(f"  Raw friends field: {repr(str(user['friends'])[:100])}")
    
    return users_df


def main():
    """Main execution"""
    # Get data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'Yelp_data')
    
    print("="*60)
    print("TESTING FRIENDS FIELD PARSING")
    print("="*60)
    print(f"Data directory: {data_dir}")
    
    # Load and inspect sample users
    sample_users = load_user_sample(data_dir, n_samples=5)
    inspect_friends_field(sample_users)
    
    # Analyze all users
    users_df = analyze_friends_field(data_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
