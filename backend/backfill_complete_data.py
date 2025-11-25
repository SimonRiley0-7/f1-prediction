"""
Complete F1 Data Backfill System (2010-2025)
Fills ALL missing qualifying, pit stops, and lap times data
Uploads directly to Supabase via Prisma
"""
import asyncio
import requests
import time
from prisma import Prisma
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from collections import defaultdict

class F1DataBackfill:
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
            'errors': 0
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
            print("\n✅ Disconnected from database")
    
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
        
        # Group by season and round
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
        
        return list(unique_races.values())
    
    async def check_existing_data(self, season, round_num):
        """Check what data already exists for a race"""
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
            'has_laptimes': laptime_count > 0,
            'quali_count': quali_count,
            'pitstop_count': pitstop_count,
            'laptime_count': laptime_count
        }
    
    def fetch_qualifying_ergast(self, season, round_num):
        """Fetch qualifying data from Ergast API"""
        url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/qualifying.json"
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None
            
            data = response.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            
            if not races or not races[0].get("QualifyingResults"):
                return None
            
            return races[0]["QualifyingResults"]
        except Exception as e:
            print(f"      ❌ Ergast qualifying error: {e}")
            return None
    
    def fetch_pitstops_ergast(self, season, round_num):
        """Fetch pit stop data from Ergast API"""
        url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/pitstops.json?limit=200"
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None
            
            data = response.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            
            if not races or not races[0].get("PitStops"):
                return None
            
            return races[0]["PitStops"]
        except Exception as e:
            print(f"      ❌ Ergast pitstops error: {e}")
            return None
    
    def fetch_laptimes_ergast(self, season, round_num):
        """Fetch lap times from Ergast API (limited availability)"""
        # Ergast lap times are available but limited
        # We'll try to get them lap by lap (max 1000 per request)
        url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/laps.json?limit=2000"
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None
            
            data = response.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            
            if not races or not races[0].get("Laps"):
                return None
            
            # Flatten lap times
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
            
            return all_lap_times
        except Exception as e:
            print(f"      ❌ Ergast laptimes error: {e}")
            return None
    
    async def store_qualifying(self, season, round_num, qualifying_data):
        """Store qualifying data in database"""
        if not qualifying_data:
            return 0
        
        count = 0
        for q in qualifying_data:
            try:
                # Check if already exists
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
                print(f"        ⚠️  Error storing qualifying for {q['Driver']['driverId']}: {e}")
                self.stats['errors'] += 1
        
        return count
    
    async def store_pitstops(self, season, round_num, pitstop_data):
        """Store pit stop data in database"""
        if not pitstop_data:
            return 0
        
        count = 0
        for p in pitstop_data:
            try:
                # Check if already exists
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
                
                # Parse duration to milliseconds
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
                print(f"        ⚠️  Error storing pitstop for {p['driverId']}: {e}")
                self.stats['errors'] += 1
        
        return count
    
    async def store_laptimes(self, season, round_num, laptime_data):
        """Store lap time data in database"""
        if not laptime_data:
            return 0
        
        count = 0
        for lt in laptime_data:
            try:
                # Check if already exists
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
                
                # Convert time to milliseconds
                time_str = lt["time"]
                try:
                    # Format: "1:23.456" or "23.456"
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
                print(f"        ⚠️  Error storing laptime: {e}")
                self.stats['errors'] += 1
        
        return count
    
    async def backfill_race(self, race):
        """Backfill all missing data for a single race"""
        season = race['season']
        round_num = race['round']
        race_name = race['race_name']
        
        print(f"\n{'='*80}")
        print(f"📍 {season} Round {round_num}: {race_name}")
        print(f"{'='*80}")
        
        # Check existing data
        existing = await self.check_existing_data(season, round_num)
        
        print(f"   Current data: Quali={existing['quali_count']}, "
              f"Pits={existing['pitstop_count']}, Laps={existing['laptime_count']}")
        
        # Fetch and store qualifying
        if not existing['has_qualifying']:
            print(f"   🔍 Fetching qualifying data...")
            quali_data = self.fetch_qualifying_ergast(season, round_num)
            if quali_data:
                count = await self.store_qualifying(season, round_num, quali_data)
                print(f"   ✅ Added {count} qualifying records")
            else:
                print(f"   ⚠️  No qualifying data available")
        else:
            print(f"   ✓ Qualifying data already exists ({existing['quali_count']} records)")
        
        # Fetch and store pit stops
        if not existing['has_pitstops']:
            print(f"   🔍 Fetching pit stop data...")
            pitstop_data = self.fetch_pitstops_ergast(season, round_num)
            if pitstop_data:
                count = await self.store_pitstops(season, round_num, pitstop_data)
                print(f"   ✅ Added {count} pit stop records")
            else:
                print(f"   ⚠️  No pit stop data available")
        else:
            print(f"   ✓ Pit stop data already exists ({existing['pitstop_count']} records)")
        
        # Fetch and store lap times
        if not existing['has_laptimes']:
            print(f"   🔍 Fetching lap time data...")
            laptime_data = self.fetch_laptimes_ergast(season, round_num)
            if laptime_data:
                count = await self.store_laptimes(season, round_num, laptime_data)
                print(f"   ✅ Added {count} lap time records")
            else:
                print(f"   ⚠️  No lap time data available")
        else:
            print(f"   ✓ Lap time data already exists ({existing['laptime_count']} records)")
        
        # Rate limiting
        time.sleep(1)
    
    async def run_backfill(self, start_year=2010, end_year=2025):
        """Run complete backfill process"""
        await self.connect_db()
        
        print(f"\n{'='*80}")
        print(f"F1 DATA BACKFILL SYSTEM")
        print(f"{'='*80}")
        print(f"Target: {start_year}-{end_year}")
        print(f"Data types: Qualifying, Pit Stops, Lap Times")
        print(f"Destination: Supabase (via Prisma)")
        print(f"{'='*80}\n")
        
        # Get all races
        races = await self.get_all_races(start_year, end_year)
        print(f"📊 Found {len(races)} races to process\n")
        
        # Process each race
        for i, race in enumerate(races, 1):
            print(f"\n[{i}/{len(races)}]", end=" ")
            await self.backfill_race(race)
            
            # Progress update every 10 races
            if i % 10 == 0:
                print(f"\n{'─'*80}")
                print(f"Progress: {i}/{len(races)} races processed")
                print(f"Stats: +{self.stats['qualifying_added']} quali, "
                      f"+{self.stats['pitstops_added']} pits, "
                      f"+{self.stats['laptimes_added']} laps")
                print(f"{'─'*80}")
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"BACKFILL COMPLETE!")
        print(f"{'='*80}")
        print(f"✅ Qualifying records added: {self.stats['qualifying_added']:,}")
        print(f"✅ Pit stop records added: {self.stats['pitstops_added']:,}")
        print(f"✅ Lap time records added: {self.stats['laptimes_added']:,}")
        print(f"⏭️  Qualifying records skipped: {self.stats['qualifying_skipped']:,}")
        print(f"⏭️  Pit stop records skipped: {self.stats['pitstops_skipped']:,}")
        print(f"⏭️  Lap time records skipped: {self.stats['laptimes_skipped']:,}")
        print(f"❌ Errors: {self.stats['errors']:,}")
        print(f"{'='*80}\n")
        
        await self.disconnect_db()

async def main():
    backfill = F1DataBackfill()
    await backfill.run_backfill(start_year=2010, end_year=2025)

if __name__ == "__main__":
    print("\n🏎️  F1 Complete Data Backfill System")
    print("="*80)
    print("This will fill ALL missing data from 2010-2025")
    print("Data types: Qualifying, Pit Stops, Lap Times")
    print("Source: Ergast API")
    print("Destination: Supabase Database")
    print("="*80 + "\n")
    
    asyncio.run(main())
