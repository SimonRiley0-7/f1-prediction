"""
Debug Backfill 2023
Investigate why 2023 data is missing by trying all APIs for a specific race
Target: 2023 Round 3 (Australian Grand Prix)
"""
import requests
import fastf1
import pandas as pd
import os

# Setup FastF1
cache_dir = 'e:/Shivam/F1/f1-ai-predictor/.fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

SEASON = 2023
ROUND = 3
RACE_NAME = "Australian Grand Prix"

def check_ergast():
    print(f"\n🔍 Checking Ergast API for {SEASON} R{ROUND}...")
    
    # Qualifying
    url = f"http://api.jolpi.ca/ergast/f1/{SEASON}/{ROUND}/qualifying.json"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if races and races[0].get("QualifyingResults"):
            print(f"  ✅ Qualifying: Found {len(races[0]['QualifyingResults'])} records")
        else:
            print(f"  ❌ Qualifying: No data found")
    except Exception as e:
        print(f"  ❌ Qualifying Error: {e}")

    # Pitstops
    url = f"http://api.jolpi.ca/ergast/f1/{SEASON}/{ROUND}/pitstops.json?limit=100"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if races and races[0].get("PitStops"):
            print(f"  ✅ Pitstops: Found {len(races[0]['PitStops'])} records")
        else:
            print(f"  ❌ Pitstops: No data found")
    except Exception as e:
        print(f"  ❌ Pitstops Error: {e}")

def check_fastf1():
    print(f"\n🔍 Checking FastF1 API for {SEASON} R{ROUND}...")
    
    # Qualifying
    try:
        session = fastf1.get_session(SEASON, ROUND, 'Q')
        session.load()
        if not session.results.empty:
            print(f"  ✅ Qualifying: Found {len(session.results)} records")
        else:
            print(f"  ❌ Qualifying: No data found")
    except Exception as e:
        print(f"  ❌ Qualifying Error: {e}")

    # Pitstops
    try:
        session = fastf1.get_session(SEASON, ROUND, 'R')
        session.load()
        if not session.laps.empty:
            pit_stops = session.laps[session.laps['PitOutTime'].notna()]
            print(f"  ✅ Pitstops: Found {len(pit_stops)} records")
        else:
            print(f"  ❌ Pitstops: No data found")
    except Exception as e:
        print(f"  ❌ Pitstops Error: {e}")

def check_openf1():
    print(f"\n🔍 Checking OpenF1 API for {SEASON} R{ROUND}...")
    
    # Get session key
    url = f"https://api.openf1.org/v1/sessions?year={SEASON}&session_name=Race&round={ROUND}"
    try:
        resp = requests.get(url, timeout=10)
        sessions = resp.json()
        if not sessions:
            print("  ❌ Session not found")
            return
            
        session_key = sessions[0]['session_key']
        print(f"  Found session key: {session_key}")
        
        # Pitstops
        url = f"https://api.openf1.org/v1/pit?session_key={session_key}"
        resp = requests.get(url, timeout=10)
        pits = resp.json()
        if pits:
            print(f"  ✅ Pitstops: Found {len(pits)} records")
        else:
            print(f"  ❌ Pitstops: No data found")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print(f"DEBUGGING MISSING DATA FOR {SEASON} {RACE_NAME}")
    check_ergast()
    check_fastf1()
    check_openf1()
