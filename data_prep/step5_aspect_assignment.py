#!/usr/bin/env python3
"""
Step 5: Assign aspects to reviews using keyword matching
"""

import pandas as pd
import re
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Aspect keywords with priority order
ASPECT_KEYWORDS = {
    'food': [
        'food', 'taste', 'flavor', 'delicious', 'menu', 'dish', 
        'cuisine', 'meal', 'eat', 'fresh', 'ingredient'
    ],
    'service': [
        'service', 'staff', 'waiter', 'waitress', 'server', 
        'friendly', 'rude', 'attentive', 'slow', 'quick'
    ],
    'atmosphere': [
        'atmosphere', 'ambiance', 'decor', 'vibe', 'cozy', 
        'noisy', 'quiet', 'clean', 'dirty', 'interior'
    ],
    'price': [
        'price', 'cheap', 'expensive', 'value', 'worth', 
        'affordable', 'overpriced', 'reasonable', 'cost'
    ]
}

# Priority order (higher priority first)
ASPECT_PRIORITY = ['food', 'service', 'atmosphere', 'price']

def load_reviews():
    """Load filtered reviews"""
    print("Loading reviews...")
    
    df_reviews = pd.read_parquet(OUTPUT_DIR / "reviews_filtered.parquet")
    
    print(f"Loaded {len(df_reviews)} reviews")
    return df_reviews

def create_keyword_patterns():
    """Create regex patterns for each keyword"""
    patterns = {}
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        pattern_parts = [r"\b" + re.escape(k) + r"\b" for k in keywords]
        pattern = "|".join(pattern_parts)
        patterns[aspect] = pattern
    
    return patterns

def assign_aspects(df_reviews):
    """Assign aspects to all reviews"""
    print("Assigning aspects to reviews...")
    
    # Create keyword patterns
    patterns = create_keyword_patterns()
    
    df = df_reviews.copy()
    text = df["text"].fillna("").astype(str)

    counts = {}
    for aspect in ASPECT_PRIORITY:
        counts[aspect] = text.str.count(patterns[aspect], flags=re.IGNORECASE)

    counts_df = pd.DataFrame(counts)
    df["aspect"] = counts_df.idxmax(axis=1)
    
    print("Aspect assignment completed")
    return df

def analyze_aspect_distribution(df_reviews):
    """Analyze and print aspect distribution"""
    print("Analyzing aspect distribution...")
    
    aspect_counts = df_reviews['aspect'].value_counts()
    total_reviews = len(df_reviews)
    
    print("\nAspect Distribution:")
    print("====================")
    
    for aspect, count in aspect_counts.items():
        percentage = (count / total_reviews) * 100
        print(f"{aspect.capitalize()}: {count} reviews ({percentage:.1f}%)")
    
    print(f"\nTotal reviews: {total_reviews}")
    
    # Save distribution to file
    with open(OUTPUT_DIR / "aspect_distribution.txt", 'w') as f:
        f.write("Aspect Assignment Distribution\n")
        f.write("==============================\n\n")
        
        for aspect, count in aspect_counts.items():
            percentage = (count / total_reviews) * 100
            f.write(f"{aspect.capitalize()}: {count} reviews ({percentage:.1f}%)\n")
        
        f.write(f"\nTotal reviews: {total_reviews}\n")
    
    return aspect_counts

def save_results(df_reviews_with_aspects):
    """Save reviews with assigned aspects"""
    print("Saving results...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save reviews with aspects
    df_reviews_with_aspects.to_parquet(OUTPUT_DIR / "reviews_with_aspects.parquet", index=False)
    
    print(f"Saved {len(df_reviews_with_aspects)} reviews with aspect assignments")

def validate_keyword_matches(df_reviews, patterns):
    """Validate keyword matching by showing sample matches"""
    print("\nKeyword Matching Validation:")
    print("============================")
    
    for aspect in ASPECT_PRIORITY:
        pattern = re.compile(patterns[aspect], re.IGNORECASE)
        
        # Find reviews with this aspect
        aspect_reviews = df_reviews[df_reviews['aspect'] == aspect]
        
        if len(aspect_reviews) > 0:
            # Find a review that actually contains the keyword
            sample_review = None
            for _, review in aspect_reviews.iterrows():
                if pattern.search(review['text']):
                    sample_review = review
                    break
            
            if sample_review is not None:
                # Extract matching keyword
                match = pattern.search(sample_review['text'])
                keyword = match.group(0)
                
                # Show preview
                text_preview = sample_review['text'][:100] + "..." if len(sample_review['text']) > 100 else sample_review['text']
                print(f"\n{aspect.capitalize()} (matched '{keyword}'):")
                print(f"  {text_preview}")

def main():
    """Main execution function"""
    print("Starting Step 5: Aspect Assignment")
    print("=" * 50)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load reviews
    df_reviews = load_reviews()
    
    # Step 2: Create keyword patterns
    patterns = create_keyword_patterns()
    
    # Step 3: Assign aspects
    df_reviews_with_aspects = assign_aspects(df_reviews)
    
    # Step 4: Analyze distribution
    aspect_counts = analyze_aspect_distribution(df_reviews_with_aspects)
    
    # Step 5: Validate keyword matching
    validate_keyword_matches(df_reviews_with_aspects, patterns)
    
    # Step 6: Save results
    save_results(df_reviews_with_aspects)
    
    print("\nStep 5 completed successfully!")

if __name__ == "__main__":
    main()
