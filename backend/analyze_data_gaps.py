"""
Analyze F1 Data Completeness
Identifies missing qualifying, pit stop, and lap time data
"""
import asyncio
from prisma import Prisma
from collections import defaultdict

async def analyze_data_gaps():
    """Analyze data completeness across all seasons"""
    db = Prisma()
    await db.connect()
    
    print(f"\n{'='*80}")
    print(f"F1 DATA COMPLETENESS ANALYSIS")
    print(f"{'='*80}\n")
    
    # Get all unique races
    all_races = await db.race.find_many()
    
    # Group by season and round
    races_by_season = defaultdict(set)
    for race in all_races:
        races_by_season[race.season].add(race.round)
    
    print(f"📊 Total Seasons: {len(races_by_season)}")
    print(f"📊 Total Unique Races: {sum(len(rounds) for rounds in races_by_season.values())}")
    print(f"📊 Total Race Records: {len(all_races)}\n")
    
    # Analyze by recent years (2018-2025)
    recent_years = range(2018, 2026)
    
    print(f"{'='*80}")
    print(f"RECENT YEARS ANALYSIS (2018-2025)")
    print(f"{'='*80}\n")
    
    for year in recent_years:
        if year not in races_by_season:
            print(f"⚠️  {year}: No data")
            continue
        
        rounds = races_by_season[year]
        
        # Count qualifying records
        quali_count = await db.qualifying.count(where={'season': year})
        
        # Count pit stop records
        pitstop_count = await db.pitstop.count(where={'season': year})
        
        # Count lap time records
        laptime_count = await db.laptime.count(where={'season': year})
        
        # Get race count
        race_count = await db.race.count(where={'season': year})
        
        print(f"📅 {year}:")
        print(f"   Races: {len(rounds)} unique rounds, {race_count} total records")
        print(f"   Qualifying: {quali_count} records")
        print(f"   Pit Stops: {pitstop_count} records")
        print(f"   Lap Times: {laptime_count} records")
        
        # Calculate expected records (approximate)
        expected_quali = len(rounds) * 20  # ~20 drivers per race
        expected_pitstops = len(rounds) * 40  # ~2 stops per driver average
        expected_laps = len(rounds) * 1000  # ~50 laps * 20 drivers
        
        quali_pct = (quali_count / expected_quali * 100) if expected_quali > 0 else 0
        pit_pct = (pitstop_count / expected_pitstops * 100) if expected_pitstops > 0 else 0
        lap_pct = (laptime_count / expected_laps * 100) if expected_laps > 0 else 0
        
        print(f"   Completeness: Quali {quali_pct:.1f}%, Pits {pit_pct:.1f}%, Laps {lap_pct:.1f}%")
        
        # Identify missing data
        missing = []
        if quali_count == 0:
            missing.append("qualifying")
        if pitstop_count == 0:
            missing.append("pit stops")
        if laptime_count == 0:
            missing.append("lap times")
        
        if missing:
            print(f"   ⚠️  Missing: {', '.join(missing)}")
        else:
            print(f"   ✅ All data types present")
        
        print()
    
    # Detailed race-by-race analysis for 2024-2025
    print(f"{'='*80}")
    print(f"DETAILED RACE-BY-RACE ANALYSIS (2024-2025)")
    print(f"{'='*80}\n")
    
    for year in [2024, 2025]:
        if year not in races_by_season:
            continue
        
        print(f"\n📅 {year} Season:")
        print(f"{'-'*80}")
        
        for round_num in sorted(races_by_season[year]):
            # Get race info
            race_info = await db.race.find_first(
                where={'season': year, 'round': round_num}
            )
            
            if not race_info:
                continue
            
            # Count data for this specific race
            quali = await db.qualifying.count(
                where={'season': year, 'round': round_num}
            )
            pits = await db.pitstop.count(
                where={'season': year, 'round': round_num}
            )
            laps = await db.laptime.count(
                where={'season': year, 'round': round_num}
            )
            
            status_quali = "✅" if quali > 0 else "❌"
            status_pits = "✅" if pits > 0 else "❌"
            status_laps = "✅" if laps > 0 else "❌"
            
            print(f"  R{round_num:2d} {race_info.race_name:35s} | "
                  f"Quali: {status_quali} ({quali:3d}) | "
                  f"Pits: {status_pits} ({pits:3d}) | "
                  f"Laps: {status_laps} ({laps:4d})")
    
    # Summary of gaps
    print(f"\n{'='*80}")
    print(f"DATA SOURCES RECOMMENDATION")
    print(f"{'='*80}\n")
    
    print("📌 OpenF1 API (2023-2025):")
    print("   - Best for: Recent race data, real-time positions, pit stops")
    print("   - Use for: 2023, 2024, 2025 seasons")
    print()
    
    print("📌 FastF1 API (2018-2025):")
    print("   - Best for: Lap times, telemetry, detailed timing data")
    print("   - Use for: 2018-2025 seasons")
    print()
    
    print("📌 Ergast API (1950-2025):")
    print("   - Best for: Historical data, qualifying, basic pit stops")
    print("   - Use for: All seasons, especially pre-2018")
    print()
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(analyze_data_gaps())
