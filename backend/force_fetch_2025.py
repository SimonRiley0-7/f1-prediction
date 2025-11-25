"""
Force Fetch 2025 Data
Aggressively fetches all data for the 2025 season using multiple sources.
"""
import asyncio
import requests
import fastf1
import os
import pandas as pd
from prisma import Prisma
from datetime import datetime

# Setup FastF1
cache_dir = 'e:/Shivam/F1/f1-ai-predictor/.fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

async def force_fetch_2025():
    db = Prisma()
    await db.connect()
    
    print(f"\n{'='*60}")
    print(f"FORCE FETCHING 2025 DATA")
    print(f"{'='*60}\n")
    
    # Get all 2025 races
    races = await db.race.find_many(
        where={'season': 2025},
        order={'round': 'asc'}
    )
    
    # Group by round to handle duplicates
    unique_races = {}
    for r in races:
        if r.round not in unique_races:
            unique_races[r.round] = r
            
    print(f"Found {len(unique_races)} races for 2025\n")
    
    for round_num, race in unique_races.items():
        print(f"Processing Round {round_num}: {race.race_name}...")
        
        # 1. Qualifying
        q_count = await db.qualifying.count(where={'season': 2025, 'round': round_num})
        if q_count == 0:
            print("  ⚠️  Missing Qualifying - Fetching...")
            await fetch_and_store_quali(db, 2025, round_num)
        else:
            print(f"  ✅ Qualifying: {q_count} records")
            
        # 2. Pitstops
        p_count = await db.pitstop.count(where={'season': 2025, 'round': round_num})
        if p_count == 0:
            print("  ⚠️  Missing Pitstops - Fetching...")
            await fetch_and_store_pits(db, 2025, round_num)
        else:
            print(f"  ✅ Pitstops: {p_count} records")
            
        # 3. Laptimes
        l_count = await db.laptime.count(where={'season': 2025, 'round': round_num})
        if l_count == 0:
            print("  ⚠️  Missing Laptimes - Fetching...")
            await fetch_and_store_laps(db, 2025, round_num)
        else:
            print(f"  ✅ Laptimes: {l_count} records")
            
    await db.disconnect()

async def fetch_and_store_quali(db, season, round_num):
    # Try FastF1 first (most reliable for recent)
    try:
        session = fastf1.get_session(season, round_num, 'Q')
        session.load()
        if not session.results.empty:
            count = 0
            for idx, row in session.results.iterrows():
                try:
                    await db.qualifying.create(data={
                        'season': season,
                        'round': round_num,
                        'driver_id': row['Abbreviation'].lower(),
                        'position': int(row['Position']) if not pd.isna(row['Position']) else 0,
                        'q1': str(row['Q1']) if not pd.isna(row['Q1']) else None,
                        'q2': str(row['Q2']) if not pd.isna(row['Q2']) else None,
                        'q3': str(row['Q3']) if not pd.isna(row['Q3']) else None
                    })
                    count += 1
                except: pass
            print(f"    ✅ FastF1: Stored {count} records")
            return
    except Exception as e:
        print(f"    ❌ FastF1 failed: {e}")
        
    # Fallback to Ergast
    try:
        url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/qualifying.json"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if races and races[0].get("QualifyingResults"):
                count = 0
                for q in races[0]["QualifyingResults"]:
                    try:
                        await db.qualifying.create(data={
                            'season': season,
                            'round': round_num,
                            'driver_id': q["Driver"]["driverId"],
                            'position': int(q["position"]),
                            'q1': q.get("Q1"),
                            'q2': q.get("Q2"),
                            'q3': q.get("Q3")
                        })
                        count += 1
                    except: pass
                print(f"    ✅ Ergast: Stored {count} records")
    except Exception as e:
        print(f"    ❌ Ergast failed: {e}")

async def fetch_and_store_pits(db, season, round_num):
    # Try OpenF1 first (best for 2025)
    try:
        url = f"https://api.openf1.org/v1/sessions?year={season}&session_name=Race&round={round_num}"
        resp = requests.get(url, timeout=10)
        sessions = resp.json()
        if sessions:
            session_key = sessions[0]['session_key']
            url = f"https://api.openf1.org/v1/pit?session_key={session_key}"
            resp = requests.get(url, timeout=10)
            pits = resp.json()
            if pits:
                count = 0
                for p in pits:
                    try:
                        await db.pitstop.create(data={
                            'season': season,
                            'round': round_num,
                            'driver_id': str(p['driver_number']),
                            'stop': 1, # OpenF1 doesn't always give stop number, simplified
                            'lap': int(p['lap_number']),
                            'time': str(p['date']),
                            'duration': str(p['pit_duration']),
                            'duration_millis': int(float(p['pit_duration']) * 1000)
                        })
                        count += 1
                    except: pass
                print(f"    ✅ OpenF1: Stored {count} records")
                return
    except Exception as e:
        print(f"    ❌ OpenF1 failed: {e}")

    # Fallback to FastF1
    try:
        session = fastf1.get_session(season, round_num, 'R')
        session.load()
        if not session.laps.empty:
            pit_stops = session.laps[session.laps['PitOutTime'].notna()]
            if not pit_stops.empty:
                count = 0
                for idx, row in pit_stops.iterrows():
                    try:
                        duration = (row['PitInTime'] - row['PitOutTime']).total_seconds() if pd.notna(row['PitInTime']) else 0
                        await db.pitstop.create(data={
                            'season': season,
                            'round': round_num,
                            'driver_id': row['Driver'].lower(),
                            'stop': 1, # Simplified
                            'lap': int(row['LapNumber']),
                            'time': str(row['Time']),
                            'duration': str(duration),
                            'duration_millis': int(duration * 1000)
                        })
                        count += 1
                    except: pass
                print(f"    ✅ FastF1: Stored {count} records")
                return
    except Exception as e:
        print(f"    ❌ FastF1 failed: {e}")

async def fetch_and_store_laps(db, season, round_num):
    # FastF1 is best for laps
    try:
        session = fastf1.get_session(season, round_num, 'R')
        session.load()
        if not session.laps.empty:
            count = 0
            for idx, row in session.laps.iterrows():
                if pd.notna(row['LapTime']):
                    try:
                        await db.laptime.create(data={
                            'season': season,
                            'round': round_num,
                            'driver_id': row['Driver'].lower(),
                            'lap': int(row['LapNumber']),
                            'position': int(row['Position']) if pd.notna(row['Position']) else 0,
                            'time': str(row['LapTime']),
                            'time_millis': int(row['LapTime'].total_seconds() * 1000)
                        })
                        count += 1
                    except: pass
            print(f"    ✅ FastF1: Stored {count} records")
    except Exception as e:
        print(f"    ❌ FastF1 failed: {e}")

if __name__ == "__main__":
    asyncio.run(force_fetch_2025())
