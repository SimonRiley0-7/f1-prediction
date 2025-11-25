"""
Targeted Backfill for Missing Races
Specifically targets races identified as missing data in the verification report.
"""
import asyncio
import requests
import time
from prisma import Prisma
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import fastf1
import os
import pandas as pd

# Setup FastF1
cache_dir = 'e:/Shivam/F1/f1-ai-predictor/.fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# List of missing races to target (from verification report)
TARGET_RACES = [
    # 2023 Races
    (2023, 3), (2023, 4), (2023, 5), (2023, 7), (2023, 8), 
    (2023, 9), (2023, 10), (2023, 11), (2023, 12), (2023, 14),
    (2023, 16), (2023, 17),
    # 2017 Races
    (2017, 2), (2017, 7), (2017, 8), (2017, 10), (2017, 11),
    (2017, 14), (2017, 17), (2017, 18), (2017, 20)
]

class TargetedBackfill:
    def __init__(self):
        self.db = None
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        
    async def connect(self):
        self.db = Prisma()
        await self.db.connect()
        
    async def disconnect(self):
        await self.db.disconnect()
        
    async def process_race(self, season, round_num):
        print(f"\nProcessing {season} Round {round_num}...")
        
        # Check what's missing
        q_count = await self.db.qualifying.count(where={'season': season, 'round': round_num})
        p_count = await self.db.pitstop.count(where={'season': season, 'round': round_num})
        
        print(f"  Current Status: Quali={q_count}, Pits={p_count}")
        
        if q_count == 0:
            await self.fill_qualifying(season, round_num)
            
        if p_count == 0:
            await self.fill_pitstops(season, round_num)
            
    async def fill_qualifying(self, season, round_num):
        print("  🔍 Fetching Qualifying...")
        
        # Try Ergast
        url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/qualifying.json"
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if races and races[0].get("QualifyingResults"):
                    results = races[0]["QualifyingResults"]
                    print(f"    ✅ Ergast found {len(results)} records")
                    await self.store_qualifying(season, round_num, results)
                    return
        except Exception as e:
            print(f"    ❌ Ergast failed: {e}")
            
        # Try FastF1
        try:
            session = fastf1.get_session(season, round_num, 'Q')
            session.load()
            if not session.results.empty:
                results = []
                for idx, row in session.results.iterrows():
                    results.append({
                        "Driver": {"driverId": row['Abbreviation'].lower()},
                        "position": str(row['Position']) if not pd.isna(row['Position']) else "0",
                        "Q1": str(row['Q1']) if not pd.isna(row['Q1']) else None,
                        "Q2": str(row['Q2']) if not pd.isna(row['Q2']) else None,
                        "Q3": str(row['Q3']) if not pd.isna(row['Q3']) else None
                    })
                print(f"    ✅ FastF1 found {len(results)} records")
                await self.store_qualifying(season, round_num, results)
                return
        except Exception as e:
            print(f"    ❌ FastF1 failed: {e}")
            
    async def fill_pitstops(self, season, round_num):
        print("  🔍 Fetching Pitstops...")
        
        # Try Ergast
        url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/pitstops.json?limit=100"
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if races and races[0].get("PitStops"):
                    results = races[0]["PitStops"]
                    print(f"    ✅ Ergast found {len(results)} records")
                    await self.store_pitstops(season, round_num, results)
                    return
        except Exception as e:
            print(f"    ❌ Ergast failed: {e}")
            
        # Try FastF1
        try:
            session = fastf1.get_session(season, round_num, 'R')
            session.load()
            if not session.laps.empty:
                pit_stops = session.laps[session.laps['PitOutTime'].notna()]
                if not pit_stops.empty:
                    results = []
                    for driver in pit_stops['Driver'].unique():
                        driver_pits = pit_stops[pit_stops['Driver'] == driver]
                        for idx, (_, pit) in enumerate(driver_pits.iterrows(), 1):
                            results.append({
                                "driverId": driver.lower(),
                                "stop": str(idx),
                                "lap": str(int(pit['LapNumber'])),
                                "time": str(pit['Time']),
                                "duration": str(pit['PitInTime'] - pit['PitOutTime']) if pd.notna(pit['PitInTime']) else "0"
                            })
                    print(f"    ✅ FastF1 found {len(results)} records")
                    await self.store_pitstops(season, round_num, results)
                    return
        except Exception as e:
            print(f"    ❌ FastF1 failed: {e}")

    async def store_qualifying(self, season, round_num, data):
        count = 0
        for q in data:
            try:
                # Check exist
                exists = await self.db.qualifying.find_first(
                    where={'season': season, 'round': round_num, 'driver_id': q["Driver"]["driverId"]}
                )
                if not exists:
                    await self.db.qualifying.create(data={
                        'season': season,
                        'round': round_num,
                        'driver_id': q["Driver"]["driverId"],
                        'position': int(float(q["position"])),
                        'q1': q.get("Q1"),
                        'q2': q.get("Q2"),
                        'q3': q.get("Q3")
                    })
                    count += 1
            except Exception as e:
                print(f"      ⚠️ Error storing {q['Driver']['driverId']}: {e}")
        print(f"    💾 Stored {count} new records")

    async def store_pitstops(self, season, round_num, data):
        count = 0
        for p in data:
            try:
                # Check exist
                exists = await self.db.pitstop.find_first(
                    where={'season': season, 'round': round_num, 'driver_id': p["driverId"], 'stop': int(p["stop"])}
                )
                if not exists:
                    duration_str = str(p["duration"])
                    try:
                        duration_ms = int(float(duration_str) * 1000)
                    except:
                        duration_ms = None
                        
                    await self.db.pitstop.create(data={
                        'season': season,
                        'round': round_num,
                        'driver_id': p["driverId"],
                        'stop': int(p["stop"]),
                        'lap': int(p["lap"]),
                        'time': str(p["time"]),
                        'duration': duration_str,
                        'duration_millis': duration_ms
                    })
                    count += 1
            except Exception as e:
                print(f"      ⚠️ Error storing {p['driverId']}: {e}")
        print(f"    💾 Stored {count} new records")

async def main():
    backfill = TargetedBackfill()
    await backfill.connect()
    
    print(f"Targeting {len(TARGET_RACES)} missing races...")
    for season, round_num in TARGET_RACES:
        await backfill.process_race(season, round_num)
        
    await backfill.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
