import pandas as pd

from pathlib import Path

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


if __name__ == "__main__":
    path=Path('./')
    raw_data = load_data(path.joinpath("data/cybersecurity_attacks.csv"))
    date=pd.to_datetime(raw_data["Timestamp"]).to_frame(name="Timestamp")
    date["Year"]= date["Timestamp"].dt.year
    date["Month"]= date["Timestamp"].dt.month
    date["Day"]= date["Timestamp"].dt.day
    date["Hour"]= date["Timestamp"].dt.hour
    date["Minute"]= date["Timestamp"].dt.minute
    date["Second"]= date["Timestamp"].dt.second
    date["DayOfWeek"]= date["Timestamp"].dt.dayofweek
    date["IsWeekend"]= date["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
    date.style.format(precision=3, thousands=".", decimal=",").format_index(
    str.upper, axis=1)