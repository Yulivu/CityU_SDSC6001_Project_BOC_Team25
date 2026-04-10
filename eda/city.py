#!/usr/bin/env python3
"""
City Restaurant Analysis
Find cities with at least 500 restaurants and calculate total reviews
"""

import pandas as pd
import json
import os
from tqdm import tqdm
from pathlib import Path


def load_business_data(data_dir):
    """Load business data and filter for restaurants"""
    print("Loading business data...")
    
    businesses = []
    with open(os.path.join(data_dir, 'yelp_academic_dataset_business.json'), 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading businesses"):
            businesses.append(json.loads(line))
    
    business_df = pd.DataFrame(businesses)
    print(f"Loaded {len(business_df):,} total businesses")
    
    # Filter for restaurants only (based on categories)
    print("Filtering for restaurants...")
    restaurant_df = business_df[business_df['categories'].str.contains('Restaurants', case=False, na=False)].copy()
    print(f"Found {len(restaurant_df):,} restaurants")
    
    return restaurant_df


def load_review_counts(data_dir):
    """Load review data and count reviews per business"""
    print("Loading review data for counting...")
    
    # Only load business_id to count reviews efficiently
    business_ids = []
    with open(os.path.join(data_dir, 'yelp_academic_dataset_review.json'), 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting reviews"):
            review = json.loads(line)
            business_ids.append(review['business_id'])
    
    # Count reviews per business
    review_counts = pd.Series(business_ids).value_counts()
    print(f"Processed reviews for {len(review_counts):,} businesses")
    
    return review_counts


def analyze_cities_with_restaurants(restaurant_df, review_counts):
    """Analyze cities with at least 500 restaurants"""
    print("\n" + "="*60)
    print("ANALYZING CITIES WITH AT LEAST 500 RESTAURANTS")
    print("="*60)
    
    # Group by city and count restaurants
    city_stats = restaurant_df.groupby('city').agg({
        'business_id': 'count'
    }).rename(columns={'business_id': 'restaurant_count'})
    
    # Filter cities with at least 500 restaurants
    qualifying_cities = city_stats[city_stats['restaurant_count'] >= 500]
    print(f"Found {len(qualifying_cities):,} cities with at least 500 restaurants")
    
    # Add review counts
    print("Calculating total reviews per city...")
    city_reviews = []
    
    for city in tqdm(qualifying_cities.index, desc="Processing cities"):
        # Get all restaurant businesses in this city
        city_restaurants = restaurant_df[restaurant_df['city'] == city]['business_id']
        
        # Calculate total reviews for restaurants in this city
        total_reviews = 0
        for business_id in city_restaurants:
            total_reviews += review_counts.get(business_id, 0)
        
        city_reviews.append(total_reviews)
    
    # Add review counts to results
    qualifying_cities = qualifying_cities.copy()
    qualifying_cities['total_reviews'] = city_reviews
    
    # Sort by total reviews descending
    qualifying_cities = qualifying_cities.sort_values('total_reviews', ascending=False)
    
    return qualifying_cities


def generate_report(city_stats, output_path):
    """Generate detailed report"""
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CITY RESTAURANT ANALYSIS REPORT\n")
        f.write("Cities with at least 500 restaurants, sorted by total reviews\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total qualifying cities: {len(city_stats):,}\n\n")
        
        f.write("City Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'City':<25} {'Restaurants':>12} {'Total Reviews':>15} {'Reviews/Restaurant':>18}\n")
        f.write("-"*80 + "\n")
        
        for city, row in city_stats.iterrows():
            restaurants = int(row['restaurant_count'])
            reviews = int(row['total_reviews'])
            reviews_per_restaurant = reviews / restaurants if restaurants > 0 else 0
            
            f.write(f"{city:<25} {restaurants:>12,} {reviews:>15,} {reviews_per_restaurant:>18.1f}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Summary statistics
        f.write("Summary Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total restaurants across all qualifying cities: {city_stats['restaurant_count'].sum():,}\n")
        f.write(f"Total reviews across all qualifying cities: {city_stats['total_reviews'].sum():,}\n")
        f.write(f"Average reviews per restaurant: {city_stats['total_reviews'].sum() / city_stats['restaurant_count'].sum():.1f}\n")
        f.write(f"City with most restaurants: {city_stats['restaurant_count'].idxmax()} ({city_stats['restaurant_count'].max():,})\n")
        f.write(f"City with most reviews: {city_stats['total_reviews'].idxmax()} ({city_stats['total_reviews'].max():,})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Report saved to: {output_path}")


def print_console_summary(city_stats):
    """Print summary to console"""
    print("\n" + "="*80)
    print("CITY RESTAURANT ANALYSIS RESULTS")
    print("="*80)
    print(f"{'City':<25} {'Restaurants':>12} {'Total Reviews':>15} {'Reviews/Restaurant':>18}")
    print("-"*80)
    
    for city, row in city_stats.iterrows():
        restaurants = int(row['restaurant_count'])
        reviews = int(row['total_reviews'])
        reviews_per_restaurant = reviews / restaurants if restaurants > 0 else 0
        
        print(f"{city:<25} {restaurants:>12,} {reviews:>15,} {reviews_per_restaurant:>18.1f}")
    
    print("-"*80)
    print(f"Total qualifying cities: {len(city_stats):,}")
    print(f"Total restaurants: {city_stats['restaurant_count'].sum():,}")
    print(f"Total reviews: {city_stats['total_reviews'].sum():,}")


def main():
    """Main execution"""
    # Get data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'Yelp_data')
    
    # Create output directory
    output_dir = os.path.join(project_root, 'eda_output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("CITY RESTAURANT ANALYSIS")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load and process data
        restaurant_df = load_business_data(data_dir)
        review_counts = load_review_counts(data_dir)
        
        # Analyze cities
        city_stats = analyze_cities_with_restaurants(restaurant_df, review_counts)
        
        # Generate output
        print_console_summary(city_stats)
        
        report_path = os.path.join(output_dir, 'city_restaurant_analysis.txt')
        generate_report(city_stats, report_path)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Analysis failed - please check data files and try again")


if __name__ == "__main__":
    main()
