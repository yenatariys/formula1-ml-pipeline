import requests
import pandas as pd

def extract_data(years=3):
    """Extract Formula 1 race data from Ergast API for the last N years"""
    base_url = "https://ergast.com/api/f1"
    current_year = 2025
    start_year = current_year - years + 1

    all_races = []

    for year in range(start_year, current_year + 1):
        url = f"{base_url}/{year}/results.json?limit=1000"
        print(f"Fetching data for {year}...")
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            races = data["MRData"]["RaceTable"]["Races"]

            for race in races:
                race_info = race["Results"][0]  # winner
                all_races.append({
                    "season": race["season"],
                    "round": race["round"],
                    "race_name": race["raceName"],
                    "date": race["date"],
                    "winner": race_info["Driver"]["familyName"],
                    "constructor": race_info["Constructor"]["name"],
                    "laps": race_info["laps"],
                    "time": race_info["Time"]["time"] if "Time" in race_info else None
                })
        else:
            print(f"Failed to fetch {year}")

    return pd.DataFrame(all_races)