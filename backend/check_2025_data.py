"""Check 2025 data in database"""
import asyncio
from prisma import Prisma

async def check_2025_data():
    db = Prisma()
    await db.connect()
    
    # Count total 2025 race records
    total_races = await db.race.count(where={'season': 2025})
    
    # Get unique races
    all_races = await db.race.find_many(
        where={'season': 2025}
    )
    
    # Get unique races manually
    unique_races = {}
    for race in all_races:
        if race.round not in unique_races:
            unique_races[race.round] = {
                'round': race.round,
                'race_name': race.race_name,
                'date': race.date
            }
    
    races = list(unique_races.values())

    
    # Count qualifying records
    quali_count = await db.qualifying.count(where={'season': 2025})
    
    # Count pit stop records
    pitstop_count = await db.pitstop.count(where={'season': 2025})
    
    print(f"\n{'='*60}")
    print(f"2025 F1 DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total race result records: {total_races}")
    print(f"Unique races: {len(races)}")
    print(f"Qualifying records: {quali_count}")
    print(f"Pit stop records: {pitstop_count}")
    print(f"\n2025 Races in Database:")
    print(f"{'-'*60}")
    
    races_sorted = sorted(races, key=lambda x: x['round'])
    for race in races_sorted:
        print(f"  Round {race['round']:2d}: {race['race_name']:40s} ({race['date']})")
    
    print(f"{'='*60}\n")
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(check_2025_data())
