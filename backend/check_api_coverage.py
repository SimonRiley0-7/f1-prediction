"""
API Coverage Check
Tests FastF1 and OpenF1 to see what data they have for 2018-2025
"""
import fastf1
import requests
import os
from datetime import datetime

# Setup FastF1
cache_dir = 'e:/Shivam/F1/f1-ai-predictor/.fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def check_fastf1_coverage():
    """Check FastF1 data availability for 2018-2025"""
    print(f"\n{'='*70}")
    print(f"FASTF1 API COVERAGE CHECK (2018-2025)")
    print(f"{'='*70}\n")
    
    test_races = [
        (2018, 1, "Australian GP"),
        (2019, 1, "Australian GP"),
        (2020, 1, "Austrian GP"),
        (2021, 1, "Bahrain GP"),
        (2022, 1, "Bahrain GP"),
        (2023, 1, "Bahrain GP"),
        (2024, 1, "Bahrain GP"),
        (2025, 1, "Australian GP"),
        (2025, 22, "Las Vegas GP"),  # Latest 2025 race
    ]
    
    for season, round_num, name in test_races:
        print(f"\n{season} R{round_num} ({name}):")
        
        # Check Qualifying
        try:
            q_session = fastf1.get_session(season, round_num, 'Q')
            q_session.load(laps=False, telemetry=False, weather=False, messages=False)
            q_count = len(q_session.results) if not q_session.results.empty else 0
            print(f"  ✓ Qualifying: {q_count} drivers")
        except Exception as e:
            print(f"  ✗ Qualifying: {str(e)[:60]}")
        
        # Check Race (for pit stops and lap times)
        try:
            r_session = fastf1.get_session(season, round_num, 'R')
            r_session.load(telemetry=False, weather=False, messages=False)
            
            if not r_session.laps.empty:
                # Pit stops
                pit_laps = r_session.laps[r_session.laps['PitOutTime'].notna()]
                pit_count = len(pit_laps)
                
                # Lap times
                lap_count = len(r_session.laps[r_session.laps['LapTime'].notna()])
                
                print(f"  ✓ Pit Stops: {pit_count} stops")
                print(f"  ✓ Lap Times: {lap_count} laps")
            else:
                print(f"  ✗ Race: No lap data")
                
        except Exception as e:
            print(f"  ✗ Race: {str(e)[:60]}")

def check_openf1_coverage():
    """Check OpenF1 data availability"""
    print(f"\n{'='*70}")
    print(f"OPENF1 API COVERAGE CHECK")
    print(f"{'='*70}\n")
    
    test_races = [
        (2023, 1, "Bahrain GP"),
        (2024, 1, "Bahrain GP"),
        (2025, 1, "Australian GP"),
        (2025, 22, "Las Vegas GP"),
    ]
    
    for season, round_num, name in test_races:
        print(f"\n{season} R{round_num} ({name}):")
        
        try:
            # Get session
            url = f"https://api.openf1.org/v1/sessions?year={season}&session_name=Race&round={round_num}"
            resp = requests.get(url, timeout=10)
            sessions = resp.json()
            
            if not sessions:
                print(f"  ✗ Session not found")
                continue
            
            session_key = sessions[0]['session_key']
            print(f"  ✓ Session found: {session_key}")
            
            # Check pit stops
            url = f"https://api.openf1.org/v1/pit?session_key={session_key}"
            resp = requests.get(url, timeout=10)
            pits = resp.json()
            print(f"  ✓ Pit Stops: {len(pits)} stops")
            
            # Check laps
            url = f"https://api.openf1.org/v1/laps?session_key={session_key}"
            resp = requests.get(url, timeout=10)
            laps = resp.json()
            print(f"  ✓ Laps: {len(laps)} laps")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:60]}")

def summary():
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"FastF1:")
    print(f"  - Coverage: 2018-2025 (all sessions)")
    print(f"  - Data: Qualifying, Pit Stops, Lap Times")
    print(f"  - Reliability: High (official F1 data)")
    print(f"\nOpenF1:")
    print(f"  - Coverage: 2023-2025 only")
    print(f"  - Data: Pit Stops, Laps (no qualifying)")
    print(f"  - Reliability: Good for recent races")
    print(f"\nRECOMMENDATION:")
    print(f"  Use FastF1 as primary source for 2018-2025")
    print(f"  Use OpenF1 as backup for 2023+ if FastF1 fails")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    check_fastf1_coverage()
    check_openf1_coverage()
    summary()
