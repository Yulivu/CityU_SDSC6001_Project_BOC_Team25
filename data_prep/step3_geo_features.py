#!/usr/bin/env python3
"""
Step 3: Generate geographic features for businesses
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import json

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Yelp_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_SEED = 42
NUM_ZONES = 50
ZONE_EMBEDDING_DIM = 32

def load_business_data():
    """Load filtered business data"""
    print("Loading business data...")
    
    df_business = pd.read_parquet(OUTPUT_DIR / "business_filtered.parquet")
    
    # Ensure business_id is string type
    df_business['business_id'] = df_business['business_id'].astype(str)
    
    # Filter to businesses with valid coordinates
    df_business = df_business.dropna(subset=['latitude', 'longitude'])
    
    print(f"Loaded {len(df_business)} businesses with valid coordinates")
    return df_business

def load_checkin_data():
    """Load check-in data"""
    print("Loading check-in data...")
    
    checkin_file = DATA_DIR / "yelp_academic_dataset_checkin.json"
    checkins = []
    
    if not checkin_file.exists():
        print("Check-in file not found, using default values")
        return {}
    
    with open(checkin_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                checkins.append(data)
            except json.JSONDecodeError:
                continue
    
    # Process check-in data
    checkin_dict = {}
    for checkin in checkins:
        business_id = checkin.get('business_id')
        date = checkin.get('date', '')
        
        if business_id and date:
            # Count number of check-ins (dates are comma-separated)
            checkin_count = len(date.split(','))
            checkin_dict[business_id] = checkin_count
    
    print(f"Loaded check-in data for {len(checkin_dict)} businesses")
    return checkin_dict

def assign_zone_clusters(df_business):
    """Assign zone IDs using K-Means clustering per city"""
    print("Assigning zone clusters...")
    
    zone_assignments = []
    
    for city in df_business['city'].unique():
        city_businesses = df_business[df_business['city'] == city]
        
        if len(city_businesses) < NUM_ZONES:
            # If not enough businesses, assign all to same zone
            k = min(len(city_businesses), 1)
        else:
            k = NUM_ZONES
        
        # Extract coordinates
        coords = city_businesses[['latitude', 'longitude']].values
        
        # Run K-Means
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        zone_ids = kmeans.fit_predict(coords)
        
        # Create zone assignment with city prefix
        for i, (_, business) in enumerate(city_businesses.iterrows()):
            zone_id = f"{city}_zone_{zone_ids[i]}"
            zone_assignments.append({
                'business_id': str(business['business_id']),  # Ensure string type
                'city': city,
                'zone_id': zone_id,
                'cluster_id': zone_ids[i]
            })
    
    df_zones = pd.DataFrame(zone_assignments)
    print(f"Assigned zones for {len(df_zones)} businesses")
    return df_zones

def compute_checkin_density(df_business, checkin_dict):
    """Compute log-normalized check-in density"""
    print("Computing check-in density...")
    
    # Add check-in counts
    df_business['checkin_count'] = df_business['business_id'].map(checkin_dict).fillna(0)
    
    # Compute density per city
    density_data = []
    
    for city in df_business['city'].unique():
        city_businesses = df_business[df_business['city'] == city]
        
        # Get total check-ins in city
        total_checkins = city_businesses['checkin_count'].sum()
        
        # Compute normalized density (add 1 to avoid log(0))
        for _, business in city_businesses.iterrows():
            if total_checkins > 0:
                density = business['checkin_count'] / total_checkins
            else:
                density = 0
            
            # Log-normalize (add 1 to avoid log(0))
            log_density = np.log1p(density)
            
            density_data.append({
                'business_id': str(business['business_id']),  # Ensure string type
                'checkin_density': log_density
            })
    
    df_density = pd.DataFrame(density_data)
    print(f"Computed density for {len(df_density)} businesses")
    return df_density

def create_zone_embeddings(df_zones):
    """Create random zone embeddings (to be fine-tuned later)"""
    print("Creating zone embeddings...")
    
    np.random.seed(RANDOM_SEED)
    
    # Get unique zone IDs
    unique_zones = df_zones['zone_id'].unique()
    
    # Create random embeddings
    zone_embeddings = {}
    for zone_id in unique_zones:
        embedding = np.random.randn(ZONE_EMBEDDING_DIM)
        zone_embeddings[zone_id] = embedding
    
    # Add embeddings to zone data
    embedding_cols = [f'zone_embedding_{i}' for i in range(ZONE_EMBEDDING_DIM)]
    
    embeddings_list = []
    for _, row in df_zones.iterrows():
        zone_id = row['zone_id']
        embedding = zone_embeddings[zone_id]
        
        embedding_dict = {'business_id': str(row['business_id'])}  # Ensure string type
        for i, val in enumerate(embedding):
            embedding_dict[f'zone_embedding_{i}'] = val
        
        embeddings_list.append(embedding_dict)
    
    df_embeddings = pd.DataFrame(embeddings_list)
    return df_embeddings

def combine_features(df_business, df_zones, df_density, df_embeddings):
    """Combine all geographic features"""
    print("Combining features...")
    
    # Merge all features
    df_geo = df_business[['business_id', 'city']].merge(
        df_zones[['business_id', 'zone_id', 'cluster_id']], on='business_id'
    ).merge(
        df_density[['business_id', 'checkin_density']], on='business_id'
    ).merge(
        df_embeddings, on='business_id'
    )
    
    print(f"Combined features for {len(df_geo)} businesses")
    return df_geo

def save_results(df_geo):
    """Save geographic features"""
    print("Saving results...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_geo.to_parquet(OUTPUT_DIR / "business_geo_features.parquet", index=False)
    
    # Print summary statistics
    print("\nGeographic Features Summary:")
    print(f"Total businesses: {len(df_geo)}")
    print(f"Cities: {len(df_geo['city'].unique())}")
    print(f"Zones: {len(df_geo['zone_id'].unique())}")
    print(f"Check-in density range: {df_geo['checkin_density'].min():.4f} to {df_geo['checkin_density'].max():.4f}")

def main():
    """Main execution function"""
    print("Starting Step 3: Geographic Features")
    print("=" * 50)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Step 1: Load business data
    df_business = load_business_data()
    
    # Step 2: Load check-in data
    checkin_dict = load_checkin_data()
    
    # Step 3: Assign zone clusters
    df_zones = assign_zone_clusters(df_business)
    
    # Step 4: Compute check-in density
    df_density = compute_checkin_density(df_business, checkin_dict)
    
    # Step 5: Create zone embeddings
    df_embeddings = create_zone_embeddings(df_zones)
    
    # Step 6: Combine features
    df_geo = combine_features(df_business, df_zones, df_density, df_embeddings)
    
    # Step 7: Save results
    save_results(df_geo)
    
    print("\nStep 3 completed successfully!")

if __name__ == "__main__":
    main()
