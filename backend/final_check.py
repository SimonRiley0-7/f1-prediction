"""
Final Data Check
Checks for missing data in the 2010-2025 range
"""
import asyncio
from prisma import Prisma

async def check_data():
    db = Prisma()
    await db.connect()
    
    print("Checking data from 2010 to 2025...")
    print("-" * 50)
    
    races = await db.race.find_many(
        where={
            'season': {
                'gte': 2010,
                'lte': 2025
            }
        },
        order={'date': 'asc'}
    )
    
    print(f"Total Races Found: {len(races)}")
    
    missing_quali = 0
    missing_pits = 0
    missing_laps = 0
    
    for race in races:
        # Check Qualifying
        q_count = await db.qualifying.count(
            where={'season': race.season, 'round': race.round}
        )
        if q_count == 0:
            missing_quali += 1
            print(f"MISSING QUALI: {race.season} Round {race.round} - {race.race_name}")
            
        # Check Pit Stops
        p_count = await db.pitstop.count(
            where={'season': race.season, 'round': race.round}
        )
        if p_count == 0:
            # Some races genuinely have no pit stops (e.g. red flagged races like Spa 2021)
            # But we'll flag them anyway
            missing_pits += 1
            # Only print if it's not a known exception
            if p_count == 0 and race.season >= 2012: # Pit data mostly available from 2012
                 print(f"NO PIT DATA:   {race.season} Round {race.round} - {race.race_name}")

        # Check Lap Times
        l_count = await db.laptime.count(
             where={'season': race.season, 'round': race.round}
        )
        if l_count == 0:
            missing_laps += 1
            if race.season >= 2018: # Lap data mostly available from 2018 via FastF1
                print(f"NO LAP DATA:   {race.season} Round {race.round} - {race.race_name}")
    
    print("-" * 50)
    print(f"Summary (2010-2025):")
    print(f"Races with NO Qualifying: {missing_quali}")
    print(f"Races with NO Pit Stops:  {missing_pits}")
    print(f"Races with NO Lap Times:  {missing_laps}")
    print("-" * 50)
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(check_data())
