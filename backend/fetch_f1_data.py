import sys
import os
from datetime import datetime, timedelta
import requests
import asyncio
    "monza": {"lat": 45.6156, "lon": 9.2811, "timezone": "Europe/Rome"},
    "spa": {"lat": 50.4372, "lon": 5.9714, "timezone": "Europe/Brussels"},
    "monaco": {"lat": 43.7347, "lon": 7.4206, "timezone": "Europe/Monaco"},
    "interlagos": {"lat": -23.7036, "lon": -46.6997, "timezone": "America/Sao_Paulo"},
    "suzuka": {"lat": 34.8431, "lon": 136.5419, "timezone": "Asia/Tokyo"},
    "albert_park": {"lat": -37.8497, "lon": 144.9680, "timezone": "Australia/Melbourne"},
    "bahrain": {"lat": 26.0325, "lon": 50.5106, "timezone": "Asia/Bahrain"},
    "shanghai": {"lat": 31.3389, "lon": 121.2197, "timezone": "Asia/Shanghai"},
    "catalunya": {"lat": 41.5700, "lon": 2.2611, "timezone": "Europe/Madrid"},
    "hungaroring": {"lat": 47.5789, "lon": 19.2486, "timezone": "Europe/Budapest"},
    "red_bull_ring": {"lat": 47.2197, "lon": 14.7647, "timezone": "Europe/Vienna"},
    "zandvoort": {"lat": 52.3888, "lon": 4.5409, "timezone": "Europe/Amsterdam"},
    "villeneuve": {"lat": 45.5000, "lon": -73.5228, "timezone": "America/Montreal"},
    "austin": {"lat": 30.1328, "lon": -97.6411, "timezone": "America/Chicago"},
    "mexico": {"lat": 19.4042, "lon": -99.0907, "timezone": "America/Mexico_City"},
    "yas_marina": {"lat": 24.4672, "lon": 54.6031, "timezone": "Asia/Dubai"},
    "losail": {"lat": 25.4900, "lon": 51.4542, "timezone": "Asia/Qatar"},
    "jeddah": {"lat": 21.6319, "lon": 39.1044, "timezone": "Asia/Riyadh"},
    "miami": {"lat": 25.9581, "lon": -80.2389, "timezone": "America/New_York"},
    "las_vegas": {"lat": 36.1617, "lon": -115.1736, "timezone": "America/Los_Angeles"},
    "baku": {"lat": 40.3725, "lon": 49.8533, "timezone": "Asia/Baku"},
    "imola": {"lat": 44.3439, "lon": 11.7167, "timezone": "Europe/Rome"},
    "portimao": {"lat": 37.2272, "lon": -8.6267, "timezone": "Europe/Lisbon"},
    "istanbul": {"lat": 40.9517, "lon": 29.4058, "timezone": "Europe/Istanbul"},
    "singapore": {"lat": 1.2914, "lon": 103.8640, "timezone": "Asia/Singapore"},
    
    # Historic European Circuits
    "nurburgring": {"lat": 50.3356, "lon": 6.9475, "timezone": "Europe/Berlin"},
    "hockenheimring": {"lat": 49.3278, "lon": 8.5658, "timezone": "Europe/Berlin"},
    "ricard": {"lat": 43.2506, "lon": 5.7919, "timezone": "Europe/Paris"},
    "magny_cours": {"lat": 47.0644, "lon": 3.1661, "timezone": "Europe/Paris"},
    "jerez": {"lat": 36.7083, "lon": -6.0347, "timezone": "Europe/Madrid"},
    "estoril": {"lat": 38.7506, "lon": -9.3939, "timezone": "Europe/Lisbon"},
    "jarama": {"lat": 40.6211, "lon": -3.5844, "timezone": "Europe/Madrid"},
    "brands_hatch": {"lat": 51.3569, "lon": 0.2631, "timezone": "Europe/London"},
    "aintree": {"lat": 53.4775, "lon": -2.9336, "timezone": "Europe/London"},
    "donington": {"lat": 52.8306, "lon": -1.3758, "timezone": "Europe/London"},
    "valencia": {"lat": 39.4553, "lon": -0.3220, "timezone": "Europe/Madrid"},
    "anderstorp": {"lat": 57.2644, "lon": 13.6000, "timezone": "Europe/Stockholm"},
    "zolder": {"lat": 50.9889, "lon": 5.2558, "timezone": "Europe/Brussels"},
    "nivelles": {"lat": 50.5956, "lon": 4.3275, "timezone": "Europe/Brussels"},
    "dijon": {"lat": 47.3644, "lon": 5.0839, "timezone": "Europe/Paris"},
    "clermont_ferrand": {"lat": 45.8972, "lon": 3.1136, "timezone": "Europe/Paris"},
    "rouen": {"lat": 49.3331, "lon": 1.0964, "timezone": "Europe/Paris"},
    "reims": {"lat": 49.2306, "lon": 4.0019, "timezone": "Europe/Paris"},
    "avus": {"lat": 52.4833, "lon": 13.2333, "timezone": "Europe/Berlin"},
    "pescara": {"lat": 42.4611, "lon": 14.2053, "timezone": "Europe/Rome"},
    "montjuic": {"lat": 41.3644, "lon": 2.1508, "timezone": "Europe/Madrid"},
    "zeltweg": {"lat": 47.2025, "lon": 14.7558, "timezone": "Europe/Vienna"},
    
    # North American Circuits
    "watkins_glen": {"lat": 42.3369, "lon": -76.9275, "timezone": "America/New_York"},
    "long_beach": {"lat": 33.7659, "lon": -118.1731, "timezone": "America/Los_Angeles"},
    "detroit": {"lat": 42.3292, "lon": -83.0403, "timezone": "America/Detroit"},
    "dallas": {"lat": 32.7767, "lon": -96.7970, "timezone": "America/Chicago"},
    "phoenix": {"lat": 33.4484, "lon": -112.0740, "timezone": "America/Phoenix"},
    "indianapolis": {"lat": 39.7950, "lon": -86.2353, "timezone": "America/Indiana/Indianapolis"},
    "riverside": {"lat": 33.9375, "lon": -117.2731, "timezone": "America/Los_Angeles"},
    "sebring": {"lat": 27.4506, "lon": -81.3481, "timezone": "America/New_York"},
    "caesars_palace": {"lat": 36.1162, "lon": -115.1719, "timezone": "America/Los_Angeles"},
    
    # Asian Circuits
    "fuji": {"lat": 35.3606, "lon": 138.9267, "timezone": "Asia/Tokyo"},
    "sepang": {"lat": 2.7606, "lon": 101.7381, "timezone": "Asia/Kuala_Lumpur"},
    "buddh_international": {"lat": 28.3492, "lon": 77.5339, "timezone": "Asia/Kolkata"},
    "korea": {"lat": 34.7339, "lon": 126.4169, "timezone": "Asia/Seoul"},
    "aida": {"lat": 34.9156, "lon": 132.5203, "timezone": "Asia/Tokyo"},
    
    # South American Circuits
    "jacarepagua": {"lat": -22.9781, "lon": -43.3953, "timezone": "America/Sao_Paulo"},
    "buenos_aires": {"lat": -34.6947, "lon": -58.4689, "timezone": "America/Argentina/Buenos_Aires"},
    
    # African Circuits
    "kyalami": {"lat": -25.9919, "lon": 28.0719, "timezone": "Africa/Johannesburg"},
    
    # Australian Circuits
    "adelaide": {"lat": -34.9275, "lon": 138.5419, "timezone": "Australia/Adelaide"},
    
    # Street Circuits (Historic)
    "phoenix_street": {"lat": 33.4484, "lon": -112.0740, "timezone": "America/Phoenix"},
    "detroit_street": {"lat": 42.3314, "lon": -83.0458, "timezone": "America/Detroit"},
    "long_beach_street": {"lat": 33.7659, "lon": -118.1731, "timezone": "America/Los_Angeles"},
    
    # Russian Circuits
    "sochi": {"lat": 43.4057, "lon": 39.9606, "timezone": "Europe/Moscow"},
    
    # Special Cases / Temporary Circuits
    "east_london": {"lat": -33.0353, "lon": 27.9139, "timezone": "Africa/Johannesburg"},
    "mosport": {"lat": 44.0589, "lon": -78.6750, "timezone": "America/Toronto"},
    "mont_tremblant": {"lat": 46.2289, "lon": -74.5911, "timezone": "America/Montreal"},
}

# Add default timezone for circuits without specific timezone
for circuit_id, coords in CIRCUIT_COORDINATES.items():
    if "timezone" not in coords:
        coords["timezone"] = "UTC"

def get_weather_condition_from_code(weather_code):
    """Convert WMO weather code to readable condition"""
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        56: "Light freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Slight thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
    }
    return weather_codes.get(weather_code, f"Unknown ({weather_code})")

def get_weather_condition_simple(weather_code):
    """Get simplified weather condition for database storage"""
    if weather_code == 0:
        return "Clear"
    elif weather_code in [1, 2]:
        return "Partly Cloudy"
    elif weather_code == 3:
        return "Cloudy"
    elif weather_code in [45, 48]:
        return "Fog"
    elif weather_code in [51, 53, 55, 56, 57]:
        return "Drizzle"
    elif weather_code in [61, 63, 65, 66, 67, 80, 81, 82]:
        return "Rain"
    elif weather_code in [71, 73, 75, 77, 85, 86]:
        return "Snow"
    elif weather_code in [95, 96, 99]:
        return "Thunderstorm"
    else:
        return "Unknown"
    
async def get_weather_data(circuit_id, race_date, race_time="14:00:00"):
    """Fetch historical weather data for a specific race using Open-Meteo API"""
    try:
        if circuit_id not in CIRCUIT_COORDINATES:
            print(f"⚠️  No coordinates found for circuit: {circuit_id}")
            return None
        
        coords = CIRCUIT_COORDINATES[circuit_id]
        
        # Parse race date
        race_date_obj = parser.parse(race_date)
        
        # Open-Meteo has data from 1940 onwards!
        if race_date_obj.year < 1940:
            print(f"⚠️  Weather data not available for {race_date} (before 1940)")
            return None
        
        # Format date for Open-Meteo (YYYY-MM-DD)
        date_str = race_date_obj.strftime('%Y-%m-%d')
        
        # Extract hour from race time
        race_hour = int(race_time.split(':')[0]) if race_time else 14
        
        # Build Open-Meteo API URL
        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'start_date': date_str,
            'end_date': date_str,
            'timezone': coords.get('timezone', 'UTC'),
            'hourly': [
                'temperature_2m',
                'relative_humidity_2m', 
                'precipitation',
                'weather_code',
                'pressure_msl',
                'surface_pressure',
                'cloud_cover',
                'visibility',
                'wind_speed_10m',
                'wind_direction_10m'
            ]
        }
        
        # Make request to Open-Meteo
        response = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=15)
        
        if response.status_code != 200:
            print(f"Open-Meteo API error: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        
        # Check if we have hourly data
        hourly_data = data.get("hourly", {})
        if not hourly_data or not hourly_data.get("time"):
            print(f"⚠️  No hourly weather data available for {date_str}")
            return None
        
        # Find the closest hour to race time
        times = hourly_data["time"]
        target_time = f"{date_str}T{race_hour:02d}:00"
        
        hour_index = None
        for i, time_str in enumerate(times):
            if time_str.startswith(target_time):
                hour_index = i
                break
        
        # If exact hour not found, find closest hour
        if hour_index is None:
            print(f"⚠️  Exact race hour not found, using closest available time")
            hour_index = min(range(len(times)), 
                           key=lambda i: abs(parser.parse(times[i]).hour - race_hour))
        
        # Extract weather data for the specific hour
        def safe_get(data_list, index):
            try:
                return data_list[index] if data_list and index < len(data_list) else None
            except (IndexError, TypeError):
                return None
        
        temperature = safe_get(hourly_data.get("temperature_2m", []), hour_index)
        humidity = safe_get(hourly_data.get("relative_humidity_2m", []), hour_index)
        precipitation = safe_get(hourly_data.get("precipitation", []), hour_index)
        weather_code = safe_get(hourly_data.get("weather_code", []), hour_index)
        pressure = safe_get(hourly_data.get("pressure_msl", []), hour_index)
        cloud_cover = safe_get(hourly_data.get("cloud_cover", []), hour_index)
        visibility = safe_get(hourly_data.get("visibility", []), hour_index)
        wind_speed = safe_get(hourly_data.get("wind_speed_10m", []), hour_index)
        wind_direction = safe_get(hourly_data.get("wind_direction_10m", []), hour_index)
        
        # Convert wind direction from degrees to compass
        def degrees_to_compass(degrees):
            if degrees is None:
                return None
            directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            idx = round(degrees / 22.5) % 16
            return directions[idx]
        
        wind_dir_compass = degrees_to_compass(wind_direction)
        
        # Prepare weather information - ONLY include fields that exist in your database schema
        weather_info = {
            "weather_condition": get_weather_condition_simple(weather_code) if weather_code is not None else None,
            "weather_description": get_weather_condition_from_code(weather_code) if weather_code is not None else None,
            "temperature_celsius": temperature,
            "humidity_percent": float(humidity) if humidity is not None else None,  # Keep as Float, not int
            "wind_speed_kmh": round(wind_speed, 1) if wind_speed is not None else None,
            "wind_direction": wind_dir_compass,
            "precipitation_mm": precipitation if precipitation is not None else 0.0,
            "weather_fetched_at": datetime.now()
        }
        
        return weather_info
        
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

async def store_comprehensive_f1_data():
    """Fetch and store all F1 data (Races, Results, Qualifying, PitStops, Weather)"""
    db = Prisma()
    await db.connect()
    
    start_year = 2020
    current_year = 2025
    
    total_races = 0
    total_results = 0
    total_duplicates = 0
    total_errors = 0
    total_weather_fetched = 0
    failed_races = []

    # Set up requests session with retry logic
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 502, 503, 504], connect=5)
    session.mount('http://', HTTPAdapter(max_retries=retries))

    for year in range(start_year, current_year + 1):
        print(f"\n{'='*50}")
        print(f"PROCESSING YEAR: {year}")
        print(f"{'='*50}")
        
        # Get all races for the year
        url = f"http://api.jolpi.ca/ergast/f1/{year}.json?limit=1000"
        year_races_stored = 0
        
        try:
            response = session.get(url, timeout=15)
            if response.status_code != 200:
                print(f"Failed to fetch races for year {year}: {response.status_code}")
                failed_races.append((year, None))
                continue
            
            data = response.json()
            races = data["MRData"]["RaceTable"]["Races"]
            
            print(f"Found {len(races)} races for year {year}")

            for race in races:
                race_round = race["round"]
                print(f"\nProcessing: {race['raceName']} - Round {race_round}")
                
                # Get weather data for this race (WeatherAPI has data from 2008)
                weather_data = {}
                print(f"  🌤️  Fetching weather data...")
                weather_info = await get_weather_data(
                    race["Circuit"]["circuitId"],
                    race["date"],
                    race.get("time", "14:00:00Z")
                )
                if weather_info:
                    weather_data = weather_info
                    total_weather_fetched += 1
                    print(f"  ✅ Weather: {weather_info.get('weather_condition', 'Unknown')} - {weather_info.get('temperature_celsius', 'N/A')}°C")
                else:
                    print(f"  ❌ Weather data not available")
               
                
                # Get detailed results for this specific race
                results_url = f"http://api.jolpi.ca/ergast/f1/{year}/{race_round}/results.json?limit=1000"
                
                try:
                    results_response = session.get(results_url, timeout=15)
                    if results_response.status_code != 200:
                        print(f"Failed to fetch results for {race['raceName']}: {results_response.status_code}")
                        failed_races.append((year, race_round))
                        continue
                    
                    results_data = results_response.json()
                    race_results = results_data["MRData"]["RaceTable"]["Races"]
                    
                    if not race_results or not race_results[0].get("Results"):
                        print(f"No results found for {race['raceName']} in {year}")
                        continue
                    
                    race_with_results = race_results[0]
                    results = race_with_results["Results"]
                    
                    print(f"Found {len(results)} driver results for this race")
                    
                    # Store each result
                    for result in results:
                        try:
                            # Check for duplicates
                            existing_result = await db.race.find_first(
                                where={
                                    "season": int(race["season"]),
                                    "round": int(race["round"]),
                                    "driver_id": result["Driver"]["driverId"]
                                }
                            )
                            
                            if existing_result:
                                print(f"  ⚠️  Duplicate skipped: {result['Driver']['familyName']} in {race['raceName']}")
                                total_duplicates += 1
                                continue
                            
                            # Prepare comprehensive race result data
                            race_data = {
                                "race_name": race["raceName"],
                                "season": int(race["season"]),
                                "round": int(race["round"]),
                                "circuit": race["Circuit"]["circuitName"],
                                "circuit_id": race["Circuit"]["circuitId"],
                                "country": race["Circuit"]["Location"]["country"],
                                "locality": race["Circuit"]["Location"]["locality"],
                                "date": race["date"],
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
                                # Weather information (enhanced with WeatherAPI data)
                                **weather_data
                            }
                            
                            # Create the race result record
                            result_record = await db.race.create(data=race_data)
                            total_results += 1
                            print(f"  ✅ Stored: {result['Driver']['familyName']} - P{result['position']} - {result['Constructor']['name']}")
                            
                        except Exception as e:
                            print(f"  ❌ Error storing result for {result['Driver']['familyName']}: {e}")
                            total_errors += 1
                            continue
                    
                    # --- Fetch Qualifying Data ---
                    print(f"  ⏱️  Fetching Qualifying data...")
                    quali_url = f"http://api.jolpi.ca/ergast/f1/{year}/{race_round}/qualifying.json"
                    try:
                        quali_response = session.get(quali_url, timeout=15)
                        if quali_response.status_code == 200:
                            quali_data = quali_response.json()["MRData"]["RaceTable"]["Races"]
                            if quali_data:
                                quali_results = quali_data[0]["QualifyingResults"]
                                for q in quali_results:
                                    try:
                                        await db.qualifying.create(data={
                                            "season": int(race["season"]),
                                            "round": int(race["round"]),
                                            "driver_id": q["Driver"]["driverId"],
                                            "position": int(q["position"]),
                                            "q1": q.get("Q1"),
                                            "q2": q.get("Q2"),
                                            "q3": q.get("Q3")
                                        })
                                    except Exception as e:
                                        # Ignore duplicates or errors
                                        pass
                                print(f"  ✅ Stored {len(quali_results)} qualifying records")
                    except Exception as e:
                        print(f"  ⚠️  Qualifying fetch failed: {e}")

                    # --- Fetch Pit Stop Data ---
                    print(f"  🛑  Fetching Pit Stop data...")
                    pit_url = f"http://api.jolpi.ca/ergast/f1/{year}/{race_round}/pitstops.json?limit=100"
                    try:
                        pit_response = session.get(pit_url, timeout=15)
                        if pit_response.status_code == 200:
                            pit_data = pit_response.json()["MRData"]["RaceTable"]["Races"]
                            if pit_data:
                                pit_stops = pit_data[0]["PitStops"]
                                for p in pit_stops:
                                    try:
                                        await db.pitstop.create(data={
                                            "season": int(race["season"]),
                                            "round": int(race["round"]),
                                            "driver_id": p["driverId"],
                                            "stop": int(p["stop"]),
                                            "lap": int(p["lap"]),
                                            "time": p["time"],
                                            "duration": p["duration"],
                                            "duration_millis": int(p["duration"]) if p["duration"].isdigit() else None # Basic check
                                        })
                                    except Exception as e:
                                        pass
                                print(f"  ✅ Stored {len(pit_stops)} pit stops")
                    except Exception as e:
                        print(f"  ⚠️  Pit Stop fetch failed: {e}")

                    year_races_stored += 1
                    total_races += 1
                    print(f"✓ Completed race: {race['raceName']} - {len(results)} results processed")
                    
                    # Delay between races to respect API limits
                    time.sleep(3)
                
                except Exception as e:
                    print(f"✗ Error processing race {race['raceName']}: {e}")
                    total_errors += 1
                    failed_races.append((year, race_round))
                    continue
            
            print(f"\nYear {year} summary: {year_races_stored} races processed")
            
            # Delay between years
            time.sleep(5)
        
        except Exception as e:
            print(f"Error fetching data for year {year}: {e}")
            total_errors += 1
            failed_races.append((year, None))

    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total races processed: {total_races}")
    print(f"Total results stored: {total_results}")
    print(f"Total weather records fetched: {total_weather_fetched}")
    print(f"Total duplicates skipped: {total_duplicates}")
    print(f"Total errors: {total_errors}")
    print(f"Average results per race: {total_results/total_races if total_races > 0 else 0:.1f}")
    if failed_races:
        print(f"Failed races (retry if needed): {failed_races}")
    
    # await db.disconnect() # Removed to keep connection open for subsequent logic
    print("Updating existing races with weather data using Open-Meteo API...")
    
    # Get all races without weather data (from 1940 onwards for Open-Meteo)
    races_without_weather = await db.race.find_many(
        where={
            "season": {"gte": 1940},
            "weather_condition": None
        },
        distinct=["season", "round", "circuit_id"],
        select={
            "season": True,
            "round": True,
            "circuit_id": True,
            "date": True,
            "time": True,
            "race_name": True
        }
    )
    
    print(f"Found {len(races_without_weather)} races without weather data")
    
    updated_count = 0
    for race in races_without_weather:
        print(f"Updating weather for {race['race_name']} {race['season']}")
        
        weather_info = await get_weather_data(
            race["circuit_id"],
            race["date"],
            race.get("time", "14:00:00Z")
        )
        
        if weather_info:
            # Update all results for this race
            await db.race.update_many(
                where={
                    "season": race["season"],
                    "round": race["round"]
                },
                data=weather_info
            )
            updated_count += 1
            print(f"  ✅ Updated: {weather_info.get('weather_condition', 'Unknown')} - {weather_info.get('temperature_celsius', 'N/A')}°C")
        else:
            print(f"  ❌ Weather data not available")
        
        # Small delay to be respectful (though Open-Meteo has no limits)
        await asyncio.sleep(0.1)
    
    print(f"Updated {updated_count} races with weather data")
    await db.disconnect()

async def backfill_weather_data_by_year(start_year=1950, end_year=None):
    """Backfill weather data for specific years using Open-Meteo"""
    db = Prisma()
    await db.connect()
    
    if end_year is None:
        end_year = datetime.now().year
    
    print(f"Backfilling weather data for years {start_year} to {end_year} using Open-Meteo")
    
    for year in range(start_year, end_year + 1):
        print(f"\n🌤️  Processing weather for year {year}")
        
        # Get all unique races for this year without weather data
        races = await db.race.find_many(
            where={
                "season": year,
                "weather_condition": None
            },
            distinct=["season", "round", "circuit_id"],
            select={
                "season": True,
                "round": True,
                "circuit_id": True,
                "date": True,
                "time": True,
                "race_name": True
            }
        )
        
        print(f"Found {len(races)} races in {year} without weather data")
        
        for race in races:
            print(f"  Fetching weather for {race['race_name']}")
            
            weather_info = await get_weather_data(
                race["circuit_id"],
                race["date"],
                race.get("time", "14:00:00Z")
            )
            
            if weather_info:
                # Update all results for this race
                updated = await db.race.update_many(
                    where={
                        "season": race["season"],
                        "round": race["round"]
                    },
                    data=weather_info
                )
                print(f"    ✅ Updated {updated} records: {weather_info.get('weather_condition', 'Unknown')} - {weather_info.get('temperature_celsius', 'N/A')}°C")
            else:
                print(f"    ❌ Weather data not available")
            
            # Small delay between requests (optional with Open-Meteo)
            await asyncio.sleep(0.1)
    
    await db.disconnect()

if __name__ == "__main__":
    print("F1 Weather Data Script - WeatherAPI Version")
    print("=" * 50)
    print("Please ensure you have:")
    print("1. Set your WeatherAPI key in WEATHER_API_KEY")
    print("2. WeatherAPI provides historical data from 2008 onwards")
    print("3. Free tier: 1M calls/month, Premium: more historical data")
    print("=" * 50)
    asyncio.run(store_comprehensive_f1_data())
    
    # For updating existing data with weather (uncomment to run)    
    # asyncio.run(update_existing_races_with_weather())
    
    # For backfilling specific years (uncomment to run)
    # asyncio.run(backfill_weather_data_by_year(2008, 2024))
    
    print("Script ready! Uncomment the appropriate function call above to run.")
    print("\nAvailable functions:")
    print("- store_comprehensive_f1_data(): Full data import from 1950 with weather from 2008+")
    print("- update_existing_races_with_weather(): Add weather to existing races")
    print("- backfill_weather_data_by_year(start, end): Backfill weather for specific years")