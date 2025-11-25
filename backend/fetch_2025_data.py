"""
Fetch 2025 F1 Data from Ergast API
Simple script to fetch the latest 2025 race results
"""
import os
import asyncio
import requests
import time
from dotenv import load_dotenv
from prisma import Prisma
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

async def fetch_2025_data():
    """Fetch all 2025 F1 race data"""
    db = Prisma()
    await db.connect()
    print("✅ Connected to database\n")
    
    year = 2025
    
    # Setup session with retry logic
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    
    print(f"{'='*60}")
    print(f"FETCHING 2025 F1 DATA")
    print(f"{'='*60}\n")
    
    # Get all races for 2025
    url = f"http://api.jolpi.ca/ergast/f1/{year}.json?limit=1000"
    
    try:
        response = session.get(url, timeout=15)
        if response.status_code != 200:
            print(f"❌ Failed to fetch races for {year}: {response.status_code}")
            return
        
        data = response.json()
        races = data["MRData"]["RaceTable"]["Races"]
        
        print(f"Found {len(races)} races for {year}\n")
        
        total_results = 0
        total_duplicates = 0
        
        for race in races:
            race_round = race["round"]
            race_name = race["raceName"]
            race_date = race["date"]
            
            print(f"\n{'='*60}")
            print(f"Processing: {race_name} - Round {race_round} ({race_date})")
            print(f"{'='*60}")
            
            # Get detailed results for this race
            results_url = f"http://api.jolpi.ca/ergast/f1/{year}/{race_round}/results.json?limit=1000"
            
            try:
                results_response = session.get(results_url, timeout=15)
                if results_response.status_code != 200:
                    print(f"⚠️  Failed to fetch results: {results_response.status_code}")
                    continue
                
                results_data = results_response.json()
                race_results = results_data["MRData"]["RaceTable"]["Races"]
                
                if not race_results or not race_results[0].get("Results"):
                    print(f"⚠️  No results found for this race")
                    continue
                
                results = race_results[0]["Results"]
                print(f"Found {len(results)} driver results\n")
                
                # Store each result
                for result in results:
                    try:
                        # Check for duplicates
                        existing_result = await db.race.find_first(
                            where={
                                "season": year,
                                "round": int(race_round),
                                "driver_id": result["Driver"]["driverId"]
                            }
                        )
                        
                        if existing_result:
                            print(f"  ⚠️  Duplicate skipped: {result['Driver']['familyName']}")
                            total_duplicates += 1
                            continue
                        
                        # Prepare race result data
                        race_data = {
                            "race_name": race_name,
                            "season": year,
                            "round": int(race_round),
                            "circuit": race["Circuit"]["circuitName"],
                            "circuit_id": race["Circuit"]["circuitId"],
                            "country": race["Circuit"]["Location"]["country"],
                            "locality": race["Circuit"]["Location"]["locality"],
                            "date": race_date,
                            "time": race.get("time", ""),
                            "url": race.get("url", ""),
                            # Driver information
                            "driver_id": result["Driver"]["driverId"],
                            "driver_code": result["Driver"].get("code", ""),
                            "driver_number": result["Driver"].get("permanentNumber", ""),
                            "driver_first_name": result["Driver"]["givenName"],
                            "driver_last_name": result["Driver"]["familyName"],
                            "driver_nationality": result["Driver"]["nationality"],
                            "driver_dob": result["Driver"]["dateOfBirth"],
                            "driver_url": result["Driver"]["url"],
                            # Constructor information
                            "constructor_id": result["Constructor"]["constructorId"],
                            "constructor_name": result["Constructor"]["name"],
                            "constructor_nationality": result["Constructor"]["nationality"],
                            "constructor_url": result["Constructor"]["url"],
                            # Race result information
                            "position": int(result["position"]) if result.get("position") else None,
                            "position_text": result.get("positionText", ""),
                            "points": float(result["points"]) if result.get("points") else 0.0,
                            "grid": int(result["grid"]) if result.get("grid") else None,
                            "laps": int(result["laps"]) if result.get("laps") else 0,
                            "status": result["status"],
                            "status_id": int(result["statusId"]) if result.get("statusId") else None,
                            # Time information
                            "time_millis": int(result["Time"]["millis"]) if result.get("Time", {}).get("millis") else None,
                            "time_text": result["Time"]["time"] if result.get("Time", {}).get("time") else "",
                            # Fastest lap information
                            "fastest_lap_rank": int(result["FastestLap"]["rank"]) if result.get("FastestLap", {}).get("rank") else None,
                            "fastest_lap_lap": int(result["FastestLap"]["lap"]) if result.get("FastestLap", {}).get("lap") else None,
                            "fastest_lap_time": result["FastestLap"]["Time"]["time"] if result.get("FastestLap", {}).get("Time", {}).get("time") else "",
                            "fastest_lap_speed": float(result["FastestLap"]["AverageSpeed"]["speed"]) if result.get("FastestLap", {}).get("AverageSpeed", {}).get("speed") else None,
                            "fastest_lap_speed_units": result["FastestLap"]["AverageSpeed"]["units"] if result.get("FastestLap", {}).get("AverageSpeed", {}).get("units") else "",
                        }
                        
                        # Create the race result record
                        await db.race.create(data=race_data)
                        total_results += 1
                        print(f"  ✅ Stored: {result['Driver']['familyName']} - P{result['position']} - {result['Constructor']['name']}")
                        
                    except Exception as e:
                        print(f"  ❌ Error storing result for {result['Driver']['familyName']}: {e}")
                        continue
                
                # Fetch Qualifying Data
                print(f"\n  ⏱️  Fetching Qualifying data...")
                quali_url = f"http://api.jolpi.ca/ergast/f1/{year}/{race_round}/qualifying.json"
                try:
                    quali_response = session.get(quali_url, timeout=15)
                    if quali_response.status_code == 200:
                        quali_data = quali_response.json()["MRData"]["RaceTable"]["Races"]
                        if quali_data:
                            quali_results = quali_data[0]["QualifyingResults"]
                            quali_count = 0
                            for q in quali_results:
                                try:
                                    # Check if already exists
                                    existing = await db.qualifying.find_first(
                                        where={
                                            "season": year,
                                            "round": int(race_round),
                                            "driver_id": q["Driver"]["driverId"]
                                        }
                                    )
                                    if not existing:
                                        await db.qualifying.create(data={
                                            "season": year,
                                            "round": int(race_round),
                                            "driver_id": q["Driver"]["driverId"],
                                            "position": int(q["position"]),
                                            "q1": q.get("Q1"),
                                            "q2": q.get("Q2"),
                                            "q3": q.get("Q3")
                                        })
                                        quali_count += 1
                                except Exception:
                                    pass
                            print(f"  ✅ Stored {quali_count} qualifying records")
                except Exception as e:
                    print(f"  ⚠️  Qualifying fetch failed: {e}")
                
                # Fetch Pit Stop Data
                print(f"  🛑  Fetching Pit Stop data...")
                pit_url = f"http://api.jolpi.ca/ergast/f1/{year}/{race_round}/pitstops.json?limit=100"
                try:
                    pit_response = session.get(pit_url, timeout=15)
                    if pit_response.status_code == 200:
                        pit_data = pit_response.json()["MRData"]["RaceTable"]["Races"]
                        if pit_data:
                            pit_stops = pit_data[0]["PitStops"]
                            pit_count = 0
                            for p in pit_stops:
                                try:
                                    # Check if already exists
                                    existing = await db.pitstop.find_first(
                                        where={
                                            "season": year,
                                            "round": int(race_round),
                                            "driver_id": p["driverId"],
                                            "stop": int(p["stop"])
                                        }
                                    )
                                    if not existing:
                                        # Parse duration to milliseconds
                                        duration_str = p["duration"]
                                        try:
                                            duration_ms = int(float(duration_str) * 1000)
                                        except:
                                            duration_ms = None
                                        
                                        await db.pitstop.create(data={
                                            "season": year,
                                            "round": int(race_round),
                                            "driver_id": p["driverId"],
                                            "stop": int(p["stop"]),
                                            "lap": int(p["lap"]),
                                            "time": p["time"],
                                            "duration": duration_str,
                                            "duration_millis": duration_ms
                                        })
                                        pit_count += 1
                                except Exception:
                                    pass
                            print(f"  ✅ Stored {pit_count} pit stops")
                except Exception as e:
                    print(f"  ⚠️  Pit Stop fetch failed: {e}")
                
                print(f"\n✓ Completed race: {race_name}")
                
                # Delay between races to respect API limits
                time.sleep(2)
                
            except Exception as e:
                print(f"✗ Error processing race {race_name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total race results stored: {total_results}")
        print(f"Total duplicates skipped: {total_duplicates}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error fetching data for {year}: {e}")
    
    await db.disconnect()
    print("✅ Database disconnected")

if __name__ == "__main__":
    print("\n🏎️  F1 2025 Data Fetcher")
    print("="*60)
    print("Fetching latest 2025 race data from Ergast API...")
    print("="*60 + "\n")
    asyncio.run(fetch_2025_data())
