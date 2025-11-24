"""
Fetch Qualifying and Pit Stop data for existing races
"""
import os
import asyncio
import requests
import time
from dotenv import load_dotenv
from prisma import Prisma
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

async def fetch_qualifying_and_pitstops():
    """Fetch qualifying and pit stop data for races in database"""
    db = Prisma()
    await db.connect()
    print("✅ Connected to database\n")
    
    # Get unique races (season, round combinations)
    races = await db.race.find_many(
        distinct=['season', 'round']
    )
    
    # Sort in Python
    races.sort(key=lambda r: (r.season, r.round))
    
    print(f"Found {len(races)} unique races\n")
    
    # Setup session
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    
    quali_added = 0
    pitstop_added = 0
    
    for i, race in enumerate(races, 1):
        year = race.season
        round_num = race.round
        
        print(f"[{i}/{len(races)}] {race.race_name} ({year} R{round_num})")
        
        # Fetch Qualifying
        quali_url = f"http://api.jolpi.ca/ergast/f1/{year}/{round_num}/qualifying.json"
        try:
            quali_response = session.get(quali_url, timeout=15)
            if quali_response.status_code == 200:
                quali_data = quali_response.json()["MRData"]["RaceTable"]["Races"]
                if quali_data and quali_data[0].get("QualifyingResults"):
                    quali_results = quali_data[0]["QualifyingResults"]
                    for q in quali_results:
                        try:
                            # Check if already exists
                            existing = await db.qualifying.find_first(
                                where={
                                    "season": year,
                                    "round": round_num,
                                    "driver_id": q["Driver"]["driverId"]
                                }
                            )
                            if not existing:
                                await db.qualifying.create(data={
                                    "season": year,
                                    "round": round_num,
                                    "driver_id": q["Driver"]["driverId"],
                                    "position": int(q["position"]),
                                    "q1": q.get("Q1"),
                                    "q2": q.get("Q2"),
                                    "q3": q.get("Q3")
                                })
                                quali_added += 1
                        except Exception as e:
                            pass  # Skip duplicates
                    print(f"  ✅ Qualifying: {len(quali_results)} records")
        except Exception as e:
            print(f"  ⚠️  Qualifying fetch failed: {e}")
        
        # Fetch Pit Stops
        pit_url = f"http://api.jolpi.ca/ergast/f1/{year}/{round_num}/pitstops.json?limit=100"
        try:
            pit_response = session.get(pit_url, timeout=15)
            if pit_response.status_code == 200:
                pit_data = pit_response.json()["MRData"]["RaceTable"]["Races"]
                if pit_data and pit_data[0].get("PitStops"):
                    pit_stops = pit_data[0]["PitStops"]
                    for p in pit_stops:
                        try:
                            # Check if already exists
                            existing = await db.pitstop.find_first(
                                where={
                                    "season": year,
                                    "round": round_num,
                                    "driver_id": p["driverId"],
                                    "stop": int(p["stop"])
                                }
                            )
                            if not existing:
                                # Parse duration to milliseconds
                                duration_str = p["duration"]
                                try:
                                    duration_ms = int(float(duration_str) * 1000)
                                except:
                                    duration_ms = None
                                
                                await db.pitstop.create(data={
                                    "season": year,
                                    "round": round_num,
                                    "driver_id": p["driverId"],
                                    "stop": int(p["stop"]),
                                    "lap": int(p["lap"]),
                                    "time": p["time"],
                                    "duration": duration_str,
                                    "duration_millis": duration_ms
                                })
                                pitstop_added += 1
                        except Exception as e:
                            pass  # Skip duplicates
                    print(f"  ✅ Pit Stops: {len(pit_stops)} records")
        except Exception as e:
            print(f"  ⚠️  Pit Stop fetch failed: {e}")
        
        # Rate limiting
        if i % 10 == 0:
            print(f"\n  Progress: {i}/{len(races)} races processed\n")
            time.sleep(2)
        else:
            time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE!")
    print(f"{'='*60}")
    print(f"Qualifying records added: {quali_added:,}")
    print(f"Pit stop records added: {pitstop_added:,}")
    
    await db.disconnect()

if __name__ == "__main__":
    print("F1 Qualifying & Pit Stop Data Collection")
    print("="*60)
    asyncio.run(fetch_qualifying_and_pitstops())
