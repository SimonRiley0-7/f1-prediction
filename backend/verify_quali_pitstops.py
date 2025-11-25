"""
Verify Qualifying and Pitstop Data (2010-2025)
Focuses specifically on data completeness for these tables.
Handles potential duplicate race entries by grouping by season/round.
"""
import asyncio
from prisma import Prisma
from collections import defaultdict

async def verify_data():
    db = Prisma()
    await db.connect()
    
    # Fetch all races
    races = await db.race.find_many(
        where={
            'season': {
                'gte': 2010,
                'lte': 2025
            }
        }
    )
    
    # Group unique races
    unique_races = defaultdict(list)
    for race in races:
        unique_races[(race.season, race.round)].append(race)
        
    with open('verification_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"VERIFYING QUALIFYING & PITSTOP DATA (2010-2025)\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"Found {len(unique_races)} unique races across {len(races)} records.\n\n")
        
        missing_quali = []
        missing_pits = []
        good_quali = 0
        good_pits = 0
        
        f.write("Checking each race...\n")
        
        sorted_keys = sorted(unique_races.keys())
        
        for season, round_num in sorted_keys:
            race_name = unique_races[(season, round_num)][0].race_name
            
            # Check Qualifying
            q_count = await db.qualifying.count(
                where={'season': season, 'round': round_num}
            )
            
            # Check Pitstops
            p_count = await db.pitstop.count(
                where={'season': season, 'round': round_num}
            )
            
            if q_count == 0:
                missing_quali.append(f"{season} R{round_num}: {race_name}")
            else:
                good_quali += 1
                
            if p_count == 0:
                missing_pits.append(f"{season} R{round_num}: {race_name}")
            else:
                good_pits += 1

        f.write(f"\n{'='*60}\n")
        f.write(f"RESULTS SUMMARY\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total Unique Races Checked: {len(unique_races)}\n")
        f.write(f"Races with Qualifying Data: {good_quali} ({good_quali/len(unique_races)*100:.1f}%)\n")
        f.write(f"Races with Pitstop Data:    {good_pits} ({good_pits/len(unique_races)*100:.1f}%)\n")
        f.write(f"{'='*60}\n\n")
        
        if missing_quali:
            f.write(f"⚠️  MISSING QUALIFYING DATA ({len(missing_quali)} races):\n")
            for m in missing_quali:
                f.write(f"  - {m}\n")
            f.write("\n")
            
        if missing_pits:
            f.write(f"⚠️  MISSING PITSTOP DATA ({len(missing_pits)} races):\n")
            f.write("Note: Some races (e.g. Spa 2021) genuinely have no pitstops.\n")
            for m in missing_pits:
                f.write(f"  - {m}\n")
                
    print("✅ Verification report saved to verification_report.txt")
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(verify_data())
