"""
Complete F1 Data Collection using Ergast API (2018-2025)
Ergast has the most complete historical F1 data
"""
import asyncio
import requests
from prisma import Prisma
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class ErgastDataCollector:
    def __init__(self):
        self.db = None
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        
        self.stats = {
            'races_processed': 0,
            'qualifying_added': 0,
            'pitstops_added': 0,
            'laptimes_added': 0
        }

    async def connect(self):
        self.db = Prisma()
        await self.db.connect()

    async def disconnect(self):
        await self.db.disconnect()

    async def run(self):
        print(f"\n{'='*70}")
        print(f"ERGAST API DATA COLLECTION (2018-2025)")
        print(f"{'='*70}\n")

        # Fetch all seasons and rounds from Ergast
        for year in range(2018, 2026):
            await self.process_season(year)
            
        print(f"\n{'='*70}")
        print(f"COLLECTION COMPLETE")
        print(f"{'='*70}")
        print(f"Races Processed:     {self.stats['races_processed']}")
        print(f"Qualifying Added:    {self.stats['qualifying_added']}")
        print(f"Pit Stops Added:     {self.stats['pitstops_added']}")
        print(f"Lap Times Added:     {self.stats['laptimes_added']}")
        print(f"{'='*70}\n")

    async def process_season(self, year):
        print(f"\n{'='*60}")
        print(f"SEASON {year}")
        print(f"{'='*60}")
        
        # Get race schedule for the season
        url = f"http://api.jolpi.ca/ergast/f1/{year}.json"
        try:
            resp = self.session.get(url, timeout=10)
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            
            print(f"Found {len(races)} races in {year}\n")
            
            for race in races:
                round_num = int(race["round"])
                race_name = race["raceName"]
                await self.process_race(year, round_num, race_name)
                
        except Exception as e:
            print(f"Error fetching {year} schedule: {e}")

    async def process_race(self, season, round_num, race_name):
        self.stats['races_processed'] += 1
        print(f"[{season} R{round_num}] {race_name}")
        
        # 1. Qualifying
        await self.fetch_qualifying(season, round_num, race_name)
        
        # 2. Pit Stops
        await self.fetch_pitstops(season, round_num, race_name)
        
        # 3. Lap Times
        await self.fetch_laptimes(season, round_num, race_name)

    async def fetch_qualifying(self, season, round_num, race_name):
        """Fetch qualifying data from Ergast"""
        try:
            # Check if already exists
            existing = await self.db.qualifying.count(
                where={'season': season, 'round': round_num}
            )
            
            if existing > 0:
                print(f"  ✓ Qualifying: {existing} records (exists)")
                return
            
            # Fetch from Ergast
            url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/qualifying.json"
            resp = self.session.get(url, timeout=10)
            
            if resp.status_code != 200:
                print(f"  ✗ Qualifying: API error {resp.status_code}")
                return
            
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            
            if not races or not races[0].get("QualifyingResults"):
                print(f"  ⚠ Qualifying: No data available")
                return
            
            results = races[0]["QualifyingResults"]
            count = 0
            
            for q in results:
                try:
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
                except Exception as e:
                    # Skip duplicates
                    continue
            
            self.stats['qualifying_added'] += count
            print(f"  ✓ Qualifying: Added {count} records")
            
        except Exception as e:
            print(f"  ✗ Qualifying: {str(e)[:50]}")

    async def fetch_pitstops(self, season, round_num, race_name):
        """Fetch pit stop data from Ergast"""
        try:
            # Check if already exists
            existing = await self.db.pitstop.count(
                where={'season': season, 'round': round_num}
            )
            
            if existing > 0:
                print(f"  ✓ Pit Stops: {existing} records (exists)")
                return
            
            # Fetch from Ergast
            url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/pitstops.json?limit=200"
            resp = self.session.get(url, timeout=10)
            
            if resp.status_code != 200:
                print(f"  ✗ Pit Stops: API error {resp.status_code}")
                return
            
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            
            if not races or not races[0].get("PitStops"):
                print(f"  ⚠ Pit Stops: No data available")
                return
            
            results = races[0]["PitStops"]
            count = 0
            
            for p in results:
                try:
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
                    # Skip duplicates
                    continue
            
            self.stats['pitstops_added'] += count
            print(f"  ✓ Pit Stops: Added {count} records")
            
        except Exception as e:
            print(f"  ✗ Pit Stops: {str(e)[:50]}")

    async def fetch_laptimes(self, season, round_num, race_name):
        """Fetch lap times from Ergast"""
        try:
            # Check if already exists
            existing = await self.db.laptime.count(
                where={'season': season, 'round': round_num}
            )
            
            if existing > 0:
                print(f"  ✓ Lap Times: {existing} records (exists)")
                return
            
            # Ergast lap times endpoint (limited data)
            # Note: Ergast only has lap times for some races
            url = f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/laps.json?limit=2000"
            resp = self.session.get(url, timeout=15)
            
            if resp.status_code != 200:
                print(f"  ⚠ Lap Times: Not available in Ergast")
                return
            
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            
            if not races or not races[0].get("Laps"):
                print(f"  ⚠ Lap Times: Not available in Ergast")
                return
            
            laps = races[0]["Laps"]
            count = 0
            
            for lap in laps:
                lap_num = int(lap["number"])
                for timing in lap.get("Timings", []):
                    try:
                        # Convert lap time to milliseconds
                        time_str = timing["time"]
                        parts = time_str.split(":")
                        if len(parts) == 2:
                            mins, secs = parts
                            total_ms = int(float(mins) * 60000 + float(secs) * 1000)
                        else:
                            total_ms = int(float(time_str) * 1000)
                        
                        await self.db.laptime.create(data={
                            'season': season,
                            'round': round_num,
                            'driver_id': timing["driverId"],
                            'lap': lap_num,
                            'position': int(timing["position"]),
                            'time': time_str,
                            'time_millis': total_ms
                        })
                        count += 1
                    except Exception as e:
                        continue
            
            self.stats['laptimes_added'] += count
            if count > 0:
                print(f"  ✓ Lap Times: Added {count} records")
            else:
                print(f"  ⚠ Lap Times: Not available in Ergast")
            
        except Exception as e:
            print(f"  ⚠ Lap Times: Not available in Ergast")

if __name__ == "__main__":
    collector = ErgastDataCollector()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(collector.connect())
    try:
        loop.run_until_complete(collector.run())
    finally:
        loop.run_until_complete(collector.disconnect())
