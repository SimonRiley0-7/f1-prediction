"""
Complete F1 Data Backfill (2018-2025)
Collects ALL qualifying, pit stops, and lap times with race names.
Ensures no race is missed.
"""
import asyncio
import requests
import fastf1
import pandas as pd
import os
from prisma import Prisma
from datetime import datetime

# Setup FastF1
cache_dir = 'e:/Shivam/F1/f1-ai-predictor/.fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

class CompleteBackfill:
    def __init__(self):
        self.db = None
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
        print(f"COMPLETE F1 DATA BACKFILL (2018-2025)")
        print(f"{'='*70}\n")

        # Get all unique races from 2018-2025
        races = await self.db.race.find_many(
            where={'season': {'gte': 2018, 'lte': 2025}},
            order={'season': 'asc', 'round': 'asc'}
        )
        
        # Group by season/round to get unique races
        unique_races = {}
        for race in races:
            key = (race.season, race.round)
            if key not in unique_races:
                unique_races[key] = race
        
        print(f"Found {len(unique_races)} unique races from 2018-2025\n")
        
        for (season, round_num), race in sorted(unique_races.items()):
            await self.process_race(race)
            
        print(f"\n{'='*70}")
        print(f"BACKFILL COMPLETE")
        print(f"{'='*70}")
        print(f"Races Processed:     {self.stats['races_processed']}")
        print(f"Qualifying Added:    {self.stats['qualifying_added']}")
        print(f"Pit Stops Added:     {self.stats['pitstops_added']}")
        print(f"Lap Times Added:     {self.stats['laptimes_added']}")
        print(f"{'='*70}\n")

    async def process_race(self, race):
        self.stats['races_processed'] += 1
        print(f"\n[{race.season} R{race.round}] {race.race_name}")
        
        # 1. Qualifying
        await self.fetch_qualifying(race)
        
        # 2. Pit Stops
        await self.fetch_pitstops(race)
        
        # 3. Lap Times
        await self.fetch_laptimes(race)

    async def fetch_qualifying(self, race):
        """Fetch qualifying data using FastF1 (most reliable for 2018+)"""
        try:
            # Check if already exists
            existing = await self.db.qualifying.count(
                where={'season': race.season, 'round': race.round}
            )
            
            if existing > 0:
                print(f"  ✓ Qualifying: {existing} records (already exists)")
                return
            
            # Fetch from FastF1
            session = fastf1.get_session(race.season, race.round, 'Q')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            
            if session.results.empty:
                print(f"  ⚠ Qualifying: No data available")
                return
            
            count = 0
            for _, row in session.results.iterrows():
                try:
                    # Map driver abbreviation to driver_id from race table
                    driver_match = await self.db.race.find_first(
                        where={
                            'season': race.season,
                            'round': race.round,
                            'driver_code': row['Abbreviation']
                        }
                    )
                    
                    if not driver_match:
                        continue
                    
                    await self.db.qualifying.create(data={
                        'season': race.season,
                        'round': race.round,
                        'driver_id': driver_match.driver_id,
                        'position': int(row['Position']) if pd.notna(row['Position']) else 0,
                        'q1': str(row['Q1']) if pd.notna(row['Q1']) else None,
                        'q2': str(row['Q2']) if pd.notna(row['Q2']) else None,
                        'q3': str(row['Q3']) if pd.notna(row['Q3']) else None
                    })
                    count += 1
                except Exception as e:
                    continue
            
            self.stats['qualifying_added'] += count
            print(f"  ✓ Qualifying: Added {count} records")
            
        except Exception as e:
            print(f"  ✗ Qualifying: Failed - {str(e)[:50]}")

    async def fetch_pitstops(self, race):
        """Fetch pit stop data using FastF1"""
        try:
            # Check if already exists
            existing = await self.db.pitstop.count(
                where={'season': race.season, 'round': race.round}
            )
            
            if existing > 0:
                print(f"  ✓ Pit Stops: {existing} records (already exists)")
                return
            
            # Fetch from FastF1
            session = fastf1.get_session(race.season, race.round, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            
            if session.laps.empty:
                print(f"  ⚠ Pit Stops: No lap data available")
                return
            
            # Find pit stops (laps where PitOutTime is not null)
            pit_laps = session.laps[session.laps['PitOutTime'].notna()]
            
            if pit_laps.empty:
                print(f"  ⚠ Pit Stops: No pit stops in this race")
                return
            
            count = 0
            for driver in pit_laps['Driver'].unique():
                driver_pits = pit_laps[pit_laps['Driver'] == driver].sort_values('LapNumber')
                
                # Get driver_id
                driver_match = await self.db.race.find_first(
                    where={
                        'season': race.season,
                        'round': race.round,
                        'driver_code': driver
                    }
                )
                
                if not driver_match:
                    continue
                
                for stop_num, (_, pit_lap) in enumerate(driver_pits.iterrows(), 1):
                    try:
                        # Calculate pit duration
                        if pd.notna(pit_lap['PitInTime']) and pd.notna(pit_lap['PitOutTime']):
                            duration_sec = (pit_lap['PitOutTime'] - pit_lap['PitInTime']).total_seconds()
                        else:
                            duration_sec = 0
                        
                        await self.db.pitstop.create(data={
                            'season': race.season,
                            'round': race.round,
                            'driver_id': driver_match.driver_id,
                            'stop': stop_num,
                            'lap': int(pit_lap['LapNumber']),
                            'time': str(pit_lap['Time']),
                            'duration': str(duration_sec),
                            'duration_millis': int(duration_sec * 1000)
                        })
                        count += 1
                    except:
                        continue
            
            self.stats['pitstops_added'] += count
            print(f"  ✓ Pit Stops: Added {count} records")
            
        except Exception as e:
            print(f"  ✗ Pit Stops: Failed - {str(e)[:50]}")

    async def fetch_laptimes(self, race):
        """Fetch lap times using FastF1"""
        try:
            # Check if already exists
            existing = await self.db.laptime.count(
                where={'season': race.season, 'round': race.round}
            )
            
            if existing > 0:
                print(f"  ✓ Lap Times: {existing} records (already exists)")
                return
            
            # Fetch from FastF1
            session = fastf1.get_session(race.season, race.round, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            
            if session.laps.empty:
                print(f"  ⚠ Lap Times: No lap data available")
                return
            
            count = 0
            for _, lap in session.laps.iterrows():
                if pd.isna(lap['LapTime']):
                    continue
                
                # Get driver_id
                driver_match = await self.db.race.find_first(
                    where={
                        'season': race.season,
                        'round': race.round,
                        'driver_code': lap['Driver']
                    }
                )
                
                if not driver_match:
                    continue
                
                try:
                    await self.db.laptime.create(data={
                        'season': race.season,
                        'round': race.round,
                        'driver_id': driver_match.driver_id,
                        'lap': int(lap['LapNumber']),
                        'position': int(lap['Position']) if pd.notna(lap['Position']) else 0,
                        'time': str(lap['LapTime']),
                        'time_millis': int(lap['LapTime'].total_seconds() * 1000)
                    })
                    count += 1
                except:
                    continue
            
            self.stats['laptimes_added'] += count
            print(f"  ✓ Lap Times: Added {count} records")
            
        except Exception as e:
            print(f"  ✗ Lap Times: Failed - {str(e)[:50]}")

if __name__ == "__main__":
    backfill = CompleteBackfill()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(backfill.connect())
    try:
        loop.run_until_complete(backfill.run())
    finally:
        loop.run_until_complete(backfill.disconnect())
