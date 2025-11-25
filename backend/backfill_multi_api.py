"""
Multi-API F1 Data Backfill System with Intelligent Fallback
Tries multiple APIs in sequence until data is found:
1. Ergast API (fastest, historical coverage)
2. FastF1 API (detailed, 2018+)
3. OpenF1 API (most recent, 2023+)
"""
import asyncio
import requests
import time
from prisma import Prisma
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

# Try to import FastF1 (optional)
try:
    import fastf1
    import os
    
    # Create cache directory if it doesn't exist
    cache_dir = 'e:/Shivam/F1/f1-ai-predictor/.fastf1_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    fastf1.Cache.enable_cache(cache_dir)
    FASTF1_AVAILABLE = True
    print("✅ FastF1 available")
except ImportError:
    FASTF1_AVAILABLE = False
    print("⚠️  FastF1 not available (install with: pip install fastf1)")
except Exception as e:
    FASTF1_AVAILABLE = False
    print(f"⚠️  FastF1 error: {e}")

class MultiAPIBackfill:
    def __init__(self):
        self.db = None
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        
        self.stats = {
            'qualifying_added': 0,
            'pitstops_added': 0,
            'laptimes_added': 0,
            'qualifying_skipped': 0,
            'pitstops_skipped': 0,
            'laptimes_skipped': 0,
            'errors': 0,
            'api_usage': {
                'ergast': 0,
                'fastf1': 0,
                'openf1': 0
            }
        }
    
    async def connect_db(self):
        """Connect to database"""
        self.db = Prisma()
        await self.db.connect()
        print("✅ Connected to Supabase database\n")
    
    async def disconnect_db(self):
        """Disconnect from database"""
        if self.db:
            await self.db.disconnect()
    
    async def get_all_races(self, start_year=2010, end_year=2025):
        """Get all unique races from database"""
        races = await self.db.race.find_many(
            where={
                'season': {
                    'gte': start_year,
                    'lte': end_year
                }
            }
        )
        
        unique_races = {}
        for race in races:
            key = (race.season, race.round)
            if key not in unique_races:
                unique_races[key] = {
                    'season': race.season,
                    'round': race.round,
                    'race_name': race.race_name,
                    'circuit_id': race.circuit_id,
                    'date': race.date
                }
        
        return sorted(unique_races.values(), key=lambda x: (x['season'], x['round']))
    
    async def check_existing_data(self, season, round_num):
        """Check what data already exists"""
        quali_count = await self.db.qualifying.count(
            where={'season': season, 'round': round_num}
        )
        pitstop_count = await self.db.pitstop.count(
            where={'season': season, 'round': round_num}
        )
        laptime_count = await self.db.laptime.count(
            where={'season': season, 'round': round_num}
        )
        
        return {
            'has_qualifying': quali_count > 0,
            'has_pitstops': pitstop_count > 0,
            'has_laptimes': laptime_count > 0
        }
    
    # ==================== ERGAST API ====================
    
    def fetch_qualifying_ergast(self, season, round_num):
        """Fetch qualifying from Ergast"""
        url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/qualifying.json"
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if races and races[0].get("QualifyingResults"):
                    self.stats['api_usage']['ergast'] += 1
                    return races[0]["QualifyingResults"]
        except Exception as e:
            print(f"        Ergast error: {e}")
        return None
    
    def fetch_pitstops_ergast(self, season, round_num):
        """Fetch pit stops from Ergast"""
        url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/pitstops.json?limit=200"
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if races and races[0].get("PitStops"):
                    self.stats['api_usage']['ergast'] += 1
                    return races[0]["PitStops"]
        except Exception as e:
            print(f"        Ergast error: {e}")
        return None
    
    def fetch_laptimes_ergast(self, season, round_num):
        """Fetch lap times from Ergast"""
        url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/laps.json?limit=2000"
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if races and races[0].get("Laps"):
                    self.stats['api_usage']['ergast'] += 1
                    all_lap_times = []
                    for lap in races[0]["Laps"]:
                        lap_number = int(lap["number"])
                        for timing in lap["Timings"]:
                            all_lap_times.append({
                                'lap': lap_number,
                                'driver_id': timing["driverId"],
                                'position': int(timing["position"]),
                                'time': timing["time"]
                            })
                    return all_lap_times if all_lap_times else None
        except Exception as e:
            print(f"        Ergast error: {e}")
        return None
    
    # ==================== FASTF1 API ====================
    
    def fetch_qualifying_fastf1(self, season, round_num):
        """Fetch qualifying from FastF1"""
        if not FASTF1_AVAILABLE or season < 2018:
            return None
        
        try:
            session = fastf1.get_session(season, round_num, 'Q')
            session.load()
            
            results = []
            for idx, row in session.results.iterrows():
                results.append({
                    "Driver": {"driverId": row['Abbreviation'].lower()},
                    "position": str(row['Position']) if not pd.isna(row['Position']) else "0",
                    "Q1": str(row['Q1']) if not pd.isna(row['Q1']) else None,
                    "Q2": str(row['Q2']) if not pd.isna(row['Q2']) else None,
                    "Q3": str(row['Q3']) if not pd.isna(row['Q3']) else None
                })
            
            if results:
                self.stats['api_usage']['fastf1'] += 1
                return results
        except Exception as e:
            print(f"        FastF1 error: {e}")
        return None
    
    def fetch_pitstops_fastf1(self, season, round_num):
        """Fetch pit stops from FastF1"""
        if not FASTF1_AVAILABLE or season < 2018:
            return None
        
        try:
            session = fastf1.get_session(season, round_num, 'R')
            session.load()
            
            laps = session.laps
            pit_stops = laps[laps['PitOutTime'].notna()]
            
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
            
            if results:
                self.stats['api_usage']['fastf1'] += 1
                return results
        except Exception as e:
            print(f"        FastF1 error: {e}")
        return None
    
    def fetch_laptimes_fastf1(self, season, round_num):
        """Fetch lap times from FastF1"""
        if not FASTF1_AVAILABLE or season < 2018:
            return None
        
        try:
            session = fastf1.get_session(season, round_num, 'R')
            session.load()
            
            laps = session.laps
            results = []
            
            for _, lap in laps.iterrows():
                if pd.notna(lap['LapTime']):
                    results.append({
                        'lap': int(lap['LapNumber']),
                        'driver_id': lap['Driver'].lower(),
                        'position': int(lap['Position']) if pd.notna(lap['Position']) else 0,
                        'time': str(lap['LapTime'])
                    })
            
            if results:
                self.stats['api_usage']['fastf1'] += 1
                return results
        except Exception as e:
            print(f"        FastF1 error: {e}")
        return None
    
    # ==================== OPENF1 API ====================
    
    def fetch_pitstops_openf1(self, season, round_num):
        """Fetch pit stops from OpenF1"""
        if season < 2023:
            return None
        
        try:
            # Get session key first
            url = f"https://api.openf1.org/v1/sessions?year={season}&session_name=Race&round={round_num}"
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None
            
            sessions = response.json()
            if not sessions:
                return None
            
            session_key = sessions[0]['session_key']
            
            # Get pit stops
            url = f"https://api.openf1.org/v1/pit?session_key={session_key}"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                pits = response.json()
                if pits:
                    self.stats['api_usage']['openf1'] += 1
                    results = []
                    for pit in pits:
                        results.append({
                            "driverId": str(pit.get('driver_number', '')).lower(),
                            "stop": str(pit.get('pit_duration', 0)),
                            "lap": str(pit.get('lap_number', 0)),
                            "time": str(pit.get('date', '')),
                            "duration": str(pit.get('pit_duration', 0))
                        })
                    return results if results else None
        except Exception as e:
            print(f"        OpenF1 error: {e}")
        return None
    
    # ==================== MULTI-API FETCH WITH FALLBACK ====================
    
    def fetch_qualifying_multi(self, season, round_num):
        """Try multiple APIs for qualifying data"""
        print(f"      🔍 Trying Ergast...", end=" ")
        data = self.fetch_qualifying_ergast(season, round_num)
        if data:
            print(f"✅ Found {len(data)} records")
            return data
        print("❌")
        
        if FASTF1_AVAILABLE and season >= 2018:
            print(f"      🔍 Trying FastF1...", end=" ")
            data = self.fetch_qualifying_fastf1(season, round_num)
            if data:
                print(f"✅ Found {len(data)} records")
                return data
            print("❌")
        
        return None
    
    def fetch_pitstops_multi(self, season, round_num):
        """Try multiple APIs for pit stop data"""
        print(f"      🔍 Trying Ergast...", end=" ")
        data = self.fetch_pitstops_ergast(season, round_num)
        if data:
            print(f"✅ Found {len(data)} records")
            return data
        print("❌")
        
        if FASTF1_AVAILABLE and season >= 2018:
            print(f"      🔍 Trying FastF1...", end=" ")
            data = self.fetch_pitstops_fastf1(season, round_num)
            if data:
                print(f"✅ Found {len(data)} records")
                return data
            print("❌")
        
        if season >= 2023:
            print(f"      🔍 Trying OpenF1...", end=" ")
            data = self.fetch_pitstops_openf1(season, round_num)
            if data:
                print(f"✅ Found {len(data)} records")
                return data
            print("❌")
        
        return None
    
    def fetch_laptimes_multi(self, season, round_num):
        """Try multiple APIs for lap time data"""
        print(f"      🔍 Trying Ergast...", end=" ")
        data = self.fetch_laptimes_ergast(season, round_num)
        if data:
            print(f"✅ Found {len(data)} records")
            return data
        print("❌")
        
        if FASTF1_AVAILABLE and season >= 2018:
            print(f"      🔍 Trying FastF1...", end=" ")
            data = self.fetch_laptimes_fastf1(season, round_num)
            if data:
                print(f"✅ Found {len(data)} records")
                return data
            print("❌")
        
        return None
    
    # ==================== DATABASE STORAGE ====================
    
    async def store_qualifying(self, season, round_num, qualifying_data):
        """Store qualifying data"""
        if not qualifying_data:
            return 0
        
        count = 0
        for q in qualifying_data:
            try:
                existing = await self.db.qualifying.find_first(
                    where={
                        'season': season,
                        'round': round_num,
                        'driver_id': q["Driver"]["driverId"]
                    }
                )
                
                if existing:
                    self.stats['qualifying_skipped'] += 1
                    continue
                
                await self.db.qualifying.create(data={
                    'season': season,
                    'round': round_num,
                    'driver_id': q["Driver"]["driverId"],
                    'position': int(q["position"]),
                    'q1': q.get("Q1"),
                    'q2': q.get("Q2"),
                    'q3': q.get("Q3")
                })
                count += 1
                self.stats['qualifying_added'] += 1
            except Exception as e:
                self.stats['errors'] += 1
        
        return count
    
    async def store_pitstops(self, season, round_num, pitstop_data):
        """Store pit stop data"""
        if not pitstop_data:
            return 0
        
        count = 0
        for p in pitstop_data:
            try:
                existing = await self.db.pitstop.find_first(
                    where={
                        'season': season,
                        'round': round_num,
                        'driver_id': p["driverId"],
                        'stop': int(p["stop"])
                    }
                )
                
                if existing:
                    self.stats['pitstops_skipped'] += 1
                    continue
                
                duration_str = p["duration"]
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
                    'time': p["time"],
                    'duration': duration_str,
                    'duration_millis': duration_ms
                })
                count += 1
                self.stats['pitstops_added'] += 1
            except Exception as e:
                self.stats['errors'] += 1
        
        return count
    
    async def store_laptimes(self, season, round_num, laptime_data):
        """Store lap time data"""
        if not laptime_data:
            return 0
        
        count = 0
        for lt in laptime_data:
            try:
                existing = await self.db.laptime.find_first(
                    where={
                        'season': season,
                        'round': round_num,
                        'driver_id': lt["driver_id"],
                        'lap': lt["lap"]
                    }
                )
                
                if existing:
                    self.stats['laptimes_skipped'] += 1
                    continue
                
                time_str = lt["time"]
                try:
                    parts = time_str.split(':')
                    if len(parts) == 2:
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                        time_ms = int((minutes * 60 + seconds) * 1000)
                    else:
                        seconds = float(parts[0])
                        time_ms = int(seconds * 1000)
                except:
                    time_ms = 0
                
                await self.db.laptime.create(data={
                    'season': season,
                    'round': round_num,
                    'driver_id': lt["driver_id"],
                    'lap': lt["lap"],
                    'position': lt["position"],
                    'time': time_str,
                    'time_millis': time_ms
                })
                count += 1
                self.stats['laptimes_added'] += 1
            except Exception as e:
                self.stats['errors'] += 1
        
        return count
    
    # ==================== MAIN BACKFILL LOGIC ====================
    
    async def backfill_race(self, race):
        """Backfill all missing data for a race"""
        season = race['season']
        round_num = race['round']
        race_name = race['race_name']
        
        print(f"\n{'='*80}")
        print(f"📍 {season} R{round_num}: {race_name}")
        print(f"{'='*80}")
        
        existing = await self.check_existing_data(season, round_num)
        
        # Qualifying
        if not existing['has_qualifying']:
            print(f"   📋 QUALIFYING:")
            quali_data = self.fetch_qualifying_multi(season, round_num)
            if quali_data:
                count = await self.store_qualifying(season, round_num, quali_data)
                print(f"      ✅ Stored {count} records in Supabase")
            else:
                print(f"      ❌ No data found from any API")
        else:
            print(f"   ✓ Qualifying already exists")
        
        # Pit Stops
        if not existing['has_pitstops']:
            print(f"   🛑 PIT STOPS:")
            pitstop_data = self.fetch_pitstops_multi(season, round_num)
            if pitstop_data:
                count = await self.store_pitstops(season, round_num, pitstop_data)
                print(f"      ✅ Stored {count} records in Supabase")
            else:
                print(f"      ❌ No data found from any API")
        else:
            print(f"   ✓ Pit stops already exist")
        
        # Lap Times
        if not existing['has_laptimes']:
            print(f"   ⏱️  LAP TIMES:")
            laptime_data = self.fetch_laptimes_multi(season, round_num)
            if laptime_data:
                count = await self.store_laptimes(season, round_num, laptime_data)
                print(f"      ✅ Stored {count} records in Supabase")
            else:
                print(f"      ❌ No data found from any API")
        else:
            print(f"   ✓ Lap times already exist")
        
        time.sleep(0.5)
    
    async def run_backfill(self, start_year=2010, end_year=2025):
        """Run complete backfill"""
        await self.connect_db()
        
        print(f"\n{'='*80}")
        print(f"MULTI-API F1 DATA BACKFILL SYSTEM")
        print(f"{'='*80}")
        print(f"Years: {start_year}-{end_year}")
        print(f"APIs: Ergast → FastF1 → OpenF1 (fallback chain)")
        print(f"Destination: Supabase Database")
        print(f"{'='*80}\n")
        
        races = await self.get_all_races(start_year, end_year)
        print(f"📊 Found {len(races)} races to process\n")
        
        for i, race in enumerate(races, 1):
            print(f"\n[{i}/{len(races)}]", end=" ")
            await self.backfill_race(race)
            
            if i % 10 == 0:
                print(f"\n{'─'*80}")
                print(f"Progress: {i}/{len(races)} | "
                      f"+{self.stats['qualifying_added']} quali, "
                      f"+{self.stats['pitstops_added']} pits, "
                      f"+{self.stats['laptimes_added']} laps")
                print(f"{'─'*80}")
        
        print(f"\n{'='*80}")
        print(f"BACKFILL COMPLETE!")
        print(f"{'='*80}")
        print(f"✅ Qualifying: {self.stats['qualifying_added']:,} added")
        print(f"✅ Pit Stops: {self.stats['pitstops_added']:,} added")
        print(f"✅ Lap Times: {self.stats['laptimes_added']:,} added")
        print(f"\n📊 API Usage:")
        print(f"   Ergast: {self.stats['api_usage']['ergast']} requests")
        print(f"   FastF1: {self.stats['api_usage']['fastf1']} requests")
        print(f"   OpenF1: {self.stats['api_usage']['openf1']} requests")
        print(f"{'='*80}\n")
        
        await self.disconnect_db()

if __name__ == "__main__":
    print("\n🏎️  Multi-API F1 Data Backfill System")
    print("="*80)
    print("Intelligent fallback: Ergast → FastF1 → OpenF1")
    print("="*80 + "\n")
    
    backfill = MultiAPIBackfill()
    asyncio.run(backfill.run_backfill(start_year=2010, end_year=2025))
