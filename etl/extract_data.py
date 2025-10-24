import pandas as pd
import os

def extract_data(data_dir="data/"):
    # cek file ada atau tidak
    for f in ["races.csv","results.csv","drivers.csv"]:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} tidak ditemukan")

    # contoh extract race results
    races = pd.read_csv(os.path.join(data_dir, "races.csv"))
    results = pd.read_csv(os.path.join(data_dir, "results.csv"))
    drivers = pd.read_csv(os.path.join(data_dir, "drivers.csv"))

    # join contoh
    df = results.merge(races, on="raceId") \
                .merge(drivers, on="driverId")

    # pilih kolom yang relevan
    df = df[[
        "raceId", "year", "round", "name", 
        "driverId", "surname", "position", "points"
    ]]
    return df