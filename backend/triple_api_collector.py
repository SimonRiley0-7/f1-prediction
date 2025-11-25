"""
Triple-API F1 Data Collector (2018-2025)
Uses Ergast → OpenF1 → FastF1 with intelligent fallback
Ensures maximum data coverage
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

class TripleAPICollector:
    def __init__(self):
        self.db = None
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        self.stats = {
            'races': 0,
            'qualifying': 0,
            'pitstops': 0,
            'laptimes': 0,
            'ergast_q': 0,
            'openf1_q': 0,
            'fastf1_q': 0,
            'ergast_p': 0,
            'openf1_p': 0,
            'fastf1_p': 0,
            'ergast_l': 0,
            'openf1_l': 0,
            'fastf1_l': 0
        }

    async def connect(self):
        self.db = Prisma()
        await self.db.connect()

    async def disconnect(self):
        await self.db.disconnect()

    async def run(self):
        print(f"\n{'='*70}")
        print(f"TRIPLE-API F1 DATA COLLECTION (2018-2025)")
        print(f"Ergast → OpenF1 → FastF1")
        print(f"{'='*70}\n")

        for year in range(2018, 2026):
            await self.process_season(year)
            
        self.print_stats()

    async def process_season(self, year):
        print(f"\n{'='*60}")
        print(f"SEASON {year}")
        print(f"{'='*60}")
        
        # Get race schedule from Ergast
        url = f"http://api.jolpi.ca/ergast/f1/{year}.json"
        try:
            resp = self.session.get(url, timeout=10)
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            
            print(f"Found {len(races)} races\n")
            
            for race in races:
                round_num = int(race["round"])
                race_name = race["raceName"]
                await self.process_race(year, round_num, race_name)
                
        except Exception as e:
            print(f"Error: {e}")

    async def process_race(self, season, round_num, race_name):
        self.stats['races'] += 1
        print(f"[{season} R{round_num}] {race_name}")
        
        await self.collect_qualifying(season, round_num, race_name)
        await self.collect_pitstops(season, round_num, race_name)
        await self.collect_laptimes(season, round_num, race_name)

    async def collect_qualifying(self, season, round_num, race_name):
        """Try Ergast → OpenF1 → FastF1"""
        # Check if exists
        existing = await self.db.qualifying.count(where={'season': season, 'round': round_num})
        if existing > 0:
            print(f"  ✓ Qualifying: {existing} (exists)")
            return
        
        # Try Ergast
        if await self.fetch_quali_ergast(season, round_num):
            return
        
        # Try FastF1 (OpenF1 doesn't have qualifying)
        if await self.fetch_quali_fastf1(season, round_num):
            return
        
        print(f"  ✗ Qualifying: No data found")

    async def collect_pitstops(self, season, round_num, race_name):
        """Try Ergast → OpenF1 → FastF1"""
        existing = await self.db.pitstop.count(where={'season': season, 'round': round_num})
        if existing > 0:
            print(f"  ✓ Pit Stops: {existing} (exists)")
            return
        
        # Try Ergast
        if await self.fetch_pits_ergast(season, round_num):
            return
        
        # Try OpenF1 (for 2023+)
        if season >= 2023:
            if await self.fetch_pits_openf1(season, round_num):
                return
        
        # Try FastF1
        if await self.fetch_pits_fastf1(season, round_num):
            return
        
        print(f"  ⚠ Pit Stops: No data found")

    async def collect_laptimes(self, season, round_num, race_name):
        """Try Ergast → FastF1 → OpenF1"""
        existing = await self.db.laptime.count(where={'season': season, 'round': round_num})
        if existing > 0:
            print(f"  ✓ Lap Times: {existing} (exists)")
            return
        
        # Try Ergast (limited)
        if await self.fetch_laps_ergast(season, round_num):
            return
        
        # Try FastF1
        if await self.fetch_laps_fastf1(season, round_num):
            return
        
        # Try OpenF1 (for 2023+)
        if season >= 2023:
            if await self.fetch_laps_openf1(season, round_num):
                return
        
        print(f"  ⚠ Lap Times: No data found")

    # ===== ERGAST FETCHERS =====
    async def fetch_quali_ergast(self, season, round_num):
        try:
            url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/qualifying.json"
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200: return False
            
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not races or not races[0].get("QualifyingResults"): return False
            
            count = 0
            for q in races[0]["QualifyingResults"]:
                try:
                    await self.db.qualifying.create(data={
                        'season': season, 'round': round_num,
                        'driver_id': q["Driver"]["driverId"],
                        'position': int(q["position"]),
                        'q1': q.get("Q1"), 'q2': q.get("Q2"), 'q3': q.get("Q3")
                    })
                    count += 1
                except: continue
            
            if count > 0:
                self.stats['qualifying'] += count
                self.stats['ergast_q'] += 1
                print(f"  ✓ Qualifying: {count} (Ergast)")
                return True
        except: pass
        return False

    async def fetch_pits_ergast(self, season, round_num):
        try:
            url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/pitstops.json?limit=200"
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200: return False
            
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not races or not races[0].get("PitStops"): return False
            
            count = 0
            for p in races[0]["PitStops"]:
                try:
                    duration_ms = int(float(p["duration"]) * 1000) if p.get("duration") else None
                    await self.db.pitstop.create(data={
                        'season': season, 'round': round_num,
                        'driver_id': p["driverId"], 'stop': int(p["stop"]),
                        'lap': int(p["lap"]), 'time': str(p["time"]),
                        'duration': str(p["duration"]), 'duration_millis': duration_ms
                    })
                    count += 1
                except: continue
            
            if count > 0:
                self.stats['pitstops'] += count
                self.stats['ergast_p'] += 1
                print(f"  ✓ Pit Stops: {count} (Ergast)")
                return True
        except: pass
        return False

    async def fetch_laps_ergast(self, season, round_num):
        try:
            url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/laps.json?limit=2000"
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200: return False
            
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not races or not races[0].get("Laps"): return False
            
            count = 0
            for lap in races[0]["Laps"]:
                lap_num = int(lap["number"])
                for timing in lap.get("Timings", []):
                    try:
                        time_str = timing["time"]
                        parts = time_str.split(":")
                        total_ms = int(float(parts[0]) * 60000 + float(parts[1]) * 1000) if len(parts) == 2 else int(float(time_str) * 1000)
                        
                        await self.db.laptime.create(data={
                            'season': season, 'round': round_num,
                            'driver_id': timing["driverId"], 'lap': lap_num,
                            'position': int(timing["position"]),
                            'time': time_str, 'time_millis': total_ms
                        })
                        count += 1
                    except: continue
            
            if count > 0:
                self.stats['laptimes'] += count
                self.stats['ergast_l'] += 1
                print(f"  ✓ Lap Times: {count} (Ergast)")
                return True
        except: pass
        return False

    # ===== OPENF1 FETCHERS =====
    async def fetch_pits_openf1(self, season, round_num):
        try:
            url = f"https://api.openf1.org/v1/sessions?year={season}&session_name=Race&round={round_num}"
            resp = self.session.get(url, timeout=10)
            sessions = resp.json()
            if not sessions: return False
            
            session_key = sessions[0]['session_key']
            url = f"https://api.openf1.org/v1/pit?session_key={session_key}"
            resp = self.session.get(url, timeout=10)
            pits = resp.json()
            if not pits: return False
            
            count = 0
            for p in pits:
                try:
                    await self.db.pitstop.create(data={
                        'season': season, 'round': round_num,
                        'driver_id': str(p['driver_number']), 'stop': 1,
                        'lap': int(p['lap_number']), 'time': str(p['date']),
                        'duration': str(p['pit_duration']),
                        'duration_millis': int(float(p['pit_duration']) * 1000)
                    })
                    count += 1
                except: continue
            
            if count > 0:
                self.stats['pitstops'] += count
                self.stats['openf1_p'] += 1
                print(f"  ✓ Pit Stops: {count} (OpenF1)")
                return True
        except: pass
        return False

    async def fetch_laps_openf1(self, season, round_num):
        try:
            url = f"https://api.openf1.org/v1/sessions?year={season}&session_name=Race&round={round_num}"
            resp = self.session.get(url, timeout=10)
            sessions = resp.json()
            if not sessions: return False
            
            session_key = sessions[0]['session_key']
            url = f"https://api.openf1.org/v1/laps?session_key={session_key}"
            resp = self.session.get(url, timeout=10)
            laps = resp.json()
            if not laps: return False
            
            count = 0
            for lap in laps:
                try:
                    if lap.get('lap_duration'):
                        await self.db.laptime.create(data={
                            'season': season, 'round': round_num,
                            'driver_id': str(lap['driver_number']),
                            'lap': int(lap['lap_number']), 'position': 0,
                            'time': str(lap['lap_duration']),
                            'time_millis': int(float(lap['lap_duration']) * 1000)
                        })
                        count += 1
                except: continue
            
            if count > 0:
                self.stats['laptimes'] += count
                self.stats['openf1_l'] += 1
                print(f"  ✓ Lap Times: {count} (OpenF1)")
                return True
        except: pass
        return False

    # ===== FASTF1 FETCHERS =====
    async def fetch_quali_fastf1(self, season, round_num):
        try:
            session = fastf1.get_session(season, round_num, 'Q')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            if session.results.empty: return False
            
            count = 0
            for _, row in session.results.iterrows():
                try:
                    # Find driver_id from race table
                    driver_match = await self.db.race.find_first(
                        where={'season': season, 'round': round_num, 'driver_code': row['Abbreviation']}
                    )
                    if not driver_match: continue
                    
                    await self.db.qualifying.create(data={
                        'season': season, 'round': round_num,
                        'driver_id': driver_match.driver_id,
                        'position': int(row['Position']) if pd.notna(row['Position']) else 0,
                        'q1': str(row['Q1']) if pd.notna(row['Q1']) else None,
                        'q2': str(row['Q2']) if pd.notna(row['Q2']) else None,
                        'q3': str(row['Q3']) if pd.notna(row['Q3']) else None
                    })
                    count += 1
                except: continue
            
            if count > 0:
                self.stats['qualifying'] += count
                self.stats['fastf1_q'] += 1
                print(f"  ✓ Qualifying: {count} (FastF1)")
                return True
        except: pass
        return False

    async def fetch_pits_fastf1(self, season, round_num):
        try:
            session = fastf1.get_session(season, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            if session.laps.empty: return False
            
            pit_laps = session.laps[session.laps['PitOutTime'].notna()]
            if pit_laps.empty: return False
            
            count = 0
            for driver in pit_laps['Driver'].unique():
                driver_pits = pit_laps[pit_laps['Driver'] == driver].sort_values('LapNumber')
                driver_match = await self.db.race.find_first(
                    where={'season': season, 'round': round_num, 'driver_code': driver}
                )
                if not driver_match: continue
                
                for stop_num, (_, pit) in enumerate(driver_pits.iterrows(), 1):
                    try:
                        duration = (pit['PitOutTime'] - pit['PitInTime']).total_seconds() if pd.notna(pit['PitInTime']) else 0
                        await self.db.pitstop.create(data={
                            'season': season, 'round': round_num,
                            'driver_id': driver_match.driver_id, 'stop': stop_num,
                            'lap': int(pit['LapNumber']), 'time': str(pit['Time']),
                            'duration': str(duration), 'duration_millis': int(duration * 1000)
                        })
                        count += 1
                    except: continue
            
            if count > 0:
                self.stats['pitstops'] += count
                self.stats['fastf1_p'] += 1
                print(f"  ✓ Pit Stops: {count} (FastF1)")
                return True
        except: pass
        return False

    async def fetch_laps_fastf1(self, season, round_num):
        try:
            session = fastf1.get_session(season, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            if session.laps.empty: return False
            
            count = 0
            for _, lap in session.laps.iterrows():
                if pd.isna(lap['LapTime']): continue
                
                driver_match = await self.db.race.find_first(
                    where={'season': season, 'round': round_num, 'driver_code': lap['Driver']}
                )
                if not driver_match: continue
                
                try:
                    await self.db.laptime.create(data={
                        'season': season, 'round': round_num,
                        'driver_id': driver_match.driver_id, 'lap': int(lap['LapNumber']),
                        'position': int(lap['Position']) if pd.notna(lap['Position']) else 0,
                        'time': str(lap['LapTime']),
                        'time_millis': int(lap['LapTime'].total_seconds() * 1000)
                    })
                    count += 1
                except: continue
            
            if count > 0:
                self.stats['laptimes'] += count
                self.stats['fastf1_l'] += 1
                print(f"  ✓ Lap Times: {count} (FastF1)")
                return True
        except: pass
        return False

    def print_stats(self):
        print(f"\n{'='*70}")
        print(f"COLLECTION COMPLETE")
        print(f"{'='*70}")
        print(f"Races Processed:     {self.stats['races']}")
        print(f"Qualifying Added:    {self.stats['qualifying']} (Ergast: {self.stats['ergast_q']}, FastF1: {self.stats['fastf1_q']})")
        print(f"Pit Stops Added:     {self.stats['pitstops']} (Ergast: {self.stats['ergast_p']}, OpenF1: {self.stats['openf1_p']}, FastF1: {self.stats['fastf1_p']})")
        print(f"Lap Times Added:     {self.stats['laptimes']} (Ergast: {self.stats['ergast_l']}, FastF1: {self.stats['fastf1_l']}, OpenF1: {self.stats['openf1_l']})")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    collector = TripleAPICollector()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(collector.connect())
    try:
        loop.run_until_complete(collector.run())
    finally:
        loop.run_until_complete(collector.disconnect())
