"""
Universal Backfill Script
Iterates through EVERY race from 2010-2025 and fills any missing data.
Uses a robust fallback chain: Ergast -> FastF1 -> OpenF1
"""
import asyncio
import requests
import fastf1
import pandas as pd
import os
from prisma import Prisma
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup FastF1
cache_dir = 'e:/Shivam/F1/f1-ai-predictor/.fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

class UniversalBackfill:
    def __init__(self):
        self.db = None
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    async def connect(self):
        self.db = Prisma()
        await self.db.connect()

    async def disconnect(self):
        await self.db.disconnect()

    async def run(self):
        print(f"\n{'='*60}")
        print(f"UNIVERSAL BACKFILL (2010-2025)")
        print(f"{'='*60}\n")

        # Fetch all races
        races = await self.db.race.find_many(
            where={'season': {'gte': 2010, 'lte': 2025}},
            order={'date': 'asc'}
        )
        
        print(f"Found {len(races)} races to check.\n")

        for race in races:
            await self.process_race(race)

    async def process_race(self, race):
        # Check Qualifying
        q_count = await self.db.qualifying.count(where={'season': race.season, 'round': race.round})
        if q_count == 0:
            print(f"MISSING QUALI: {race.season} R{race.round} ({race.race_name})")
            await self.fill_qualifying(race)
            
        # Check Pitstops
        p_count = await self.db.pitstop.count(where={'season': race.season, 'round': race.round})
        if p_count == 0:
            # Skip 2010 pitstops as they are generally unavailable in public APIs
            if race.season == 2010:
                pass 
            else:
                print(f"MISSING PITS:  {race.season} R{race.round} ({race.race_name})")
                await self.fill_pitstops(race)

    async def fill_qualifying(self, race):
        # 1. Try Ergast
        if await self.fetch_quali_ergast(race): return
        # 2. Try FastF1
        if await self.fetch_quali_fastf1(race): return
        print(f"  ❌ Failed to find Qualifying data for {race.season} R{race.round}")

    async def fill_pitstops(self, race):
        # 1. Try Ergast
        if await self.fetch_pits_ergast(race): return
        # 2. Try FastF1
        if await self.fetch_pits_fastf1(race): return
        # 3. Try OpenF1 (2023+)
        if race.season >= 2023:
            if await self.fetch_pits_openf1(race): return
        print(f"  ❌ Failed to find Pitstop data for {race.season} R{race.round}")

    # --- ERGAST FETCHERS ---
    async def fetch_quali_ergast(self, race):
        try:
            url = f"http://api.jolpi.ca/ergast/f1/{race.season}/{race.round}/qualifying.json"
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200: return False
            
            data = resp.json()
            races_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not races_data or not races_data[0].get("QualifyingResults"): return False
            
            results = races_data[0]["QualifyingResults"]
            count = 0
            for q in results:
                try:
                    await self.db.qualifying.create(data={
                        'season': race.season,
                        'round': race.round,
                        'driver_id': q["Driver"]["driverId"],
                        'position': int(q["position"]),
                        'q1': q.get("Q1"),
                        'q2': q.get("Q2"),
                        'q3': q.get("Q3")
                    })
                    count += 1
                except: pass
            print(f"  ✅ Ergast: Stored {count} records")
            return True
        except: return False

    async def fetch_pits_ergast(self, race):
        try:
            url = f"http://api.jolpi.ca/ergast/f1/{race.season}/{race.round}/pitstops.json?limit=100"
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200: return False
            
            data = resp.json()
            races_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not races_data or not races_data[0].get("PitStops"): return False
            
            results = races_data[0]["PitStops"]
            count = 0
            for p in results:
                try:
                    duration_str = str(p["duration"])
                    try: duration_ms = int(float(duration_str) * 1000)
                    except: duration_ms = None
                    
                    await self.db.pitstop.create(data={
                        'season': race.season,
                        'round': race.round,
                        'driver_id': p["driverId"],
                        'stop': int(p["stop"]),
                        'lap': int(p["lap"]),
                        'time': str(p["time"]),
                        'duration': duration_str,
                        'duration_millis': duration_ms
                    })
                    count += 1
                except: pass
            print(f"  ✅ Ergast: Stored {count} records")
            return True
        except: return False

    # --- FASTF1 FETCHERS ---
    async def fetch_quali_fastf1(self, race):
        try:
            session = fastf1.get_session(race.season, race.round, 'Q')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            if session.results.empty: return False
            
            count = 0
            for _, row in session.results.iterrows():
                try:
                    await self.db.qualifying.create(data={
                        'season': race.season,
                        'round': race.round,
                        'driver_id': row['Abbreviation'].lower(), # Approximate driver_id
                        'position': int(row['Position']) if pd.notna(row['Position']) else 0,
                        'q1': str(row['Q1']) if pd.notna(row['Q1']) else None,
                        'q2': str(row['Q2']) if pd.notna(row['Q2']) else None,
                        'q3': str(row['Q3']) if pd.notna(row['Q3']) else None
                    })
                    count += 1
                except: pass
            print(f"  ✅ FastF1: Stored {count} records")
            return True
        except: return False

    async def fetch_pits_fastf1(self, race):
        try:
            session = fastf1.get_session(race.season, race.round, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            if session.laps.empty: return False
            
            pit_stops = session.laps[session.laps['PitOutTime'].notna()]
            if pit_stops.empty: return False
            
            count = 0
            # Need to group by driver to determine stop number
            for driver in pit_stops['Driver'].unique():
                driver_pits = pit_stops[pit_stops['Driver'] == driver]
                for idx, (_, row) in enumerate(driver_pits.iterrows(), 1):
                    try:
                        duration = (row['PitInTime'] - row['PitOutTime']).total_seconds() if pd.notna(row['PitInTime']) else 0
                        await self.db.pitstop.create(data={
                            'season': race.season,
                            'round': race.round,
                            'driver_id': row['Driver'].lower(),
                            'stop': idx,
                            'lap': int(row['LapNumber']),
                            'time': str(row['Time']),
                            'duration': str(duration),
                            'duration_millis': int(duration * 1000)
                        })
                        count += 1
                    except: pass
            print(f"  ✅ FastF1: Stored {count} records")
            return True
        except: return False

    # --- OPENF1 FETCHERS ---
    async def fetch_pits_openf1(self, race):
        try:
            # Get session key
            url = f"https://api.openf1.org/v1/sessions?year={race.season}&session_name=Race&round={race.round}"
            resp = self.session.get(url, timeout=10)
            sessions = resp.json()
            if not sessions: return False
            
            session_key = sessions[0]['session_key']
            
            # Get pits
            url = f"https://api.openf1.org/v1/pit?session_key={session_key}"
            resp = self.session.get(url, timeout=10)
            pits = resp.json()
            if not pits: return False
            
            count = 0
            for p in pits:
                try:
                    await self.db.pitstop.create(data={
                        'season': race.season,
                        'round': race.round,
                        'driver_id': str(p['driver_number']),
                        'stop': 1, # Simplified
                        'lap': int(p['lap_number']),
                        'time': str(p['date']),
                        'duration': str(p['pit_duration']),
                        'duration_millis': int(float(p['pit_duration']) * 1000)
                    })
                    count += 1
                except: pass
            print(f"  ✅ OpenF1: Stored {count} records")
            return True
        except: return False

if __name__ == "__main__":
    backfill = UniversalBackfill()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(backfill.connect())
    try:
        loop.run_until_complete(backfill.run())
    finally:
        loop.run_until_complete(backfill.disconnect())
