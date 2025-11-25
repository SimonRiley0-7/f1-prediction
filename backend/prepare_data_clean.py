"""
F1 Data Preparation Script - Clean Version
Fetches data from Supabase and prepares it for ML training
"""
import os
import asyncio
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from dotenv import load_dotenv
from prisma import Prisma

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

async def prepare_data():
    """Prepare F1 data for machine learning"""
    db = Prisma()
    await db.connect()
    print("✅ Connected to database\n")
    
    # Fetch race data (Expanded range)
    print("📊 Fetching race data (2010-2025)...")
    races = await db.race.find_many(
        where={"season": {"gte": 2010}}
    )
    print(f"   Found {len(races)} race records\n")
    
    # Fetch qualifying data
    print("⏱️  Fetching qualifying data...")
    qualifying = await db.qualifying.find_many(
        where={"season": {"gte": 2010}}
    )
    print(f"   Found {len(qualifying)} qualifying records\n")
    
    # Fetch pit stop data
    print("🛑 Fetching pit stop data...")
    pitstops = await db.pitstop.find_many(
        where={"season": {"gte": 2010}}
    )
    print(f"   Found {len(pitstops)} pit stop records\n")
    
    # Fetch lap time data (New)
    print("🏎️  Fetching lap time data...")
    laptimes = await db.laptime.find_many(
        where={"season": {"gte": 2010}}
    )
    print(f"   Found {len(laptimes)} lap time records\n")
    
    await db.disconnect()
    
    # Convert to DataFrames
    print("🔄 Converting to DataFrames...")
    df_races = pd.DataFrame([{
        'season': r.season,
        'round': r.round,
        'race_name': r.race_name,
        'circuit_id': r.circuit_id,
        'date': r.date,
        'driver_id': r.driver_id,
        'constructor_id': r.constructor_id,
        'position': r.position if r.position else 20,
        'points': r.points,
        'grid': r.grid if r.grid else 20,
        'laps': r.laps,
        'status': r.status
    } for r in races])
    
    df_quali = pd.DataFrame([{
        'season': q.season,
        'round': q.round,
        'driver_id': q.driver_id,
        'quali_pos': q.position
    } for q in qualifying])
    
    # Process Pit Stops
    pitstop_agg = pd.DataFrame([{
        'season': p.season,
        'round': p.round,
        'driver_id': p.driver_id,
        'duration': p.duration_millis
    } for p in pitstops])
    
    if not pitstop_agg.empty:
        # Calculate avg pit duration per race/driver
        pit_stats = pitstop_agg.groupby(['season', 'round', 'driver_id']).agg({
            'duration': ['count', 'mean']
        }).reset_index()
        pit_stats.columns = ['season', 'round', 'driver_id', 'num_pit_stops', 'avg_pit_duration']
        pit_stats['avg_pit_duration'] = pit_stats['avg_pit_duration'] / 1000.0 # Convert to seconds
    else:
        pit_stats = pd.DataFrame(columns=['season', 'round', 'driver_id', 'num_pit_stops', 'avg_pit_duration'])
    
    # Process Lap Times
    lap_agg = pd.DataFrame([{
        'season': l.season,
        'round': l.round,
        'driver_id': l.driver_id,
        'time_millis': l.time_millis
    } for l in laptimes])
    
    if not lap_agg.empty:
        # Calculate race pace stats
        lap_stats = lap_agg.groupby(['season', 'round', 'driver_id'])['time_millis'].agg(['mean', 'std']).reset_index()
        lap_stats.columns = ['season', 'round', 'driver_id', 'avg_lap_time_ms', 'lap_std_ms']
        
        # Calculate race average lap time (to normalize pace)
        race_avgs = lap_agg.groupby(['season', 'round'])['time_millis'].mean().reset_index(name='race_avg_ms')
        lap_stats = lap_stats.merge(race_avgs, on=['season', 'round'])
        
        # Pace ratio (lower is better, <1 means faster than average)
        lap_stats['pace_ratio'] = lap_stats['avg_lap_time_ms'] / lap_stats['race_avg_ms']
    else:
        lap_stats = pd.DataFrame(columns=['season', 'round', 'driver_id', 'pace_ratio', 'lap_std_ms'])

    # Merge data
    print("🔗 Merging datasets...")
    df = df_races.merge(df_quali, on=['season', 'round', 'driver_id'], how='left')
    df = df.merge(pit_stats, on=['season', 'round', 'driver_id'], how='left')
    df = df.merge(lap_stats[['season', 'round', 'driver_id', 'pace_ratio', 'lap_std_ms']], 
                 on=['season', 'round', 'driver_id'], how='left')
    
    # Fill missing values
    df['quali_pos'] = df['quali_pos'].fillna(df['grid'])
    df['num_pit_stops'] = df['num_pit_stops'].fillna(0)
    df['avg_pit_duration'] = df['avg_pit_duration'].fillna(25.0) # Default pit time
    df['pace_ratio'] = df['pace_ratio'].fillna(1.05) # Slower than avg default
    df['lap_std_ms'] = df['lap_std_ms'].fillna(0)
    
    # Feature engineering
    print("⚙️  Engineering advanced features...")
    
    df = df.sort_values(['driver_id', 'season', 'round'])
    
    # 1. Driver Form (Points)
    df['driver_form'] = df.groupby('driver_id')['points'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).fillna(0)
    
    # 2. Constructor Form
    df['constructor_form'] = df.groupby('constructor_id')['points'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).fillna(0)
    
    # 3. Recent Pace (Rolling avg of pace_ratio)
    df['recent_pace'] = df.groupby('driver_id')['pace_ratio'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).fillna(1.05)
    
    # 4. Recent Consistency (Rolling avg of lap_std)
    df['recent_consistency'] = df.groupby('driver_id')['lap_std_ms'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).fillna(0)
    
    # 5. Recent Pit Form (Rolling avg of pit duration)
    df['recent_pit_form'] = df.groupby('driver_id')['avg_pit_duration'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).fillna(25.0)
    
    # Grid penalty
    df['grid_penalty'] = df['grid'] - df['quali_pos']
    
    # Track experience
    df['track_experience'] = df.groupby(['driver_id', 'circuit_id']).cumcount()
    
    # Encode categorical variables
    print("🔤 Encoding categorical variables...")
    le_driver = LabelEncoder()
    le_constructor = LabelEncoder()
    le_circuit = LabelEncoder()
    
    df['driver_encoded'] = le_driver.fit_transform(df['driver_id'])
    df['constructor_encoded'] = le_constructor.fit_transform(df['constructor_id'])
    df['circuit_encoded'] = le_circuit.fit_transform(df['circuit_id'])
    
    # Save prepared data
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'f1_race_data_prepared.csv')
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved prepared data to: {output_file}")
    print(f"   Shape: {df.shape}")
    
    # Save encoders
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(le_driver, os.path.join(models_dir, 'driver_encoder.pkl'))
    joblib.dump(le_constructor, os.path.join(models_dir, 'constructor_encoder.pkl'))
    joblib.dump(le_circuit, os.path.join(models_dir, 'circuit_encoder.pkl'))
    print(f"✅ Saved encoders to: {models_dir}\n")
    
    # Print summary
    print("="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total records: {len(df):,}")
    print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"Unique drivers: {df['driver_id'].nunique()}")
    print(f"Unique constructors: {df['constructor_id'].nunique()}")
    print(f"Unique circuits: {df['circuit_id'].nunique()}")
    print(f"\nNew Features: recent_pace, recent_consistency, recent_pit_form")
    print("="*60)
    
    return df

if __name__ == "__main__":
    print("F1 Data Preparation Script (Enhanced)")
    print("="*60 + "\n")
    asyncio.run(prepare_data())
    print("\n✅ Data preparation complete!")
