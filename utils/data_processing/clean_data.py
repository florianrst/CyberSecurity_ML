import pandas as pd
import numpy as np
from pathlib import Path
from ipaddress import IPv4Address, IPv4Network
from sklearn.preprocessing import StandardScaler

from utils.data_processing import df_ua_parser

df_dtypes={
    "Source Port":"Int32",  
    "Destination Port":"Int32",
    "Protocol":"category",
    "Packet Type":"category",
    "Traffic Type":"category",
    "Malware Indicators":"category",
    "Anomaly Scores":"Float64",
    "Alerts/Warnings":"category",
    "Attack Type":"category",
    "Attack Signature":"category",
    "Action Taken":"category",
    "Severity Level":"category",
    "Network Segment":"category",
    "Geo-location Data":"str",
    "Firewall Logs":"str",
    "IDS/IPS Alerts":"str",
    "Log Source":"str",
    "Year":"Int32",
    "Month":"Int32",
    "Day":"Int32",
    "Hour":"Int32",
    "Minute":"Int32",
    "Second":"Int32",
    "DayOfWeek":"Int32",
    "IsWeekend":"bool",
    "Browser Name":"str",
    "Browser Version":"str",
    "Browser Minor":"str",
    "Browser Patch":"str",
    "Browser Patch Minor":"str",
    "OS Name":"str",
    "OS Version":"str",
    "OS Version Minor":"str",
    "OS Version Patch":"str",
    "OS Version Patch Minor":"str",
    "Device Brand":"str",
    "Device Model":"str",
    "Device Type":"str",
    "Packet Bin":"str",
    "Packet_T":"Float64",
    "Int Source IP":"Int64",
    "Int Destination IP":"Int64",
    "Int Proxy Information":"Int64",
    "Global Source IP":"bool",
    "Global Destination IP":"bool",
    "Is Proxy":"bool"
}

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path, dtype=df_dtypes)

def clean_data(file_path):
    raw_data = load_data(file_path)
    raw_data.dtypes
    date_data=pd.to_datetime(raw_data["Timestamp"]).to_frame(name="Timestamp")
    date_data["Year"]= date_data["Timestamp"].dt.year
    date_data["Month"]= date_data["Timestamp"].dt.month
    date_data["Day"]= date_data["Timestamp"].dt.day
    date_data["Hour"]= date_data["Timestamp"].dt.hour
    date_data["Minute"]= date_data["Timestamp"].dt.minute
    date_data["Second"]= date_data["Timestamp"].dt.second
    date_data["DayOfWeek"]= date_data["Timestamp"].dt.dayofweek
    date_data["IsWeekend"]= date_data["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
    date_data=date_data.drop(columns=["Timestamp"])

    device_data = df_ua_parser(raw_data["Device Information"])
    device_data = device_data.drop(columns=["String"])

    packet_data= raw_data["Packet Length"].to_frame(name="Packet Length")
    packet_data["Packet Bin"] = packet_data["Packet Length"].apply(lambda x: "Small" if x < 420 else ("Medium" if x < 1143 else "Large"))
    scaler = StandardScaler().fit(packet_data["Packet Length"].to_frame())
    packet_data["Packet_T"] = scaler.transform(packet_data["Packet Length"].to_frame())
    packet_data = packet_data.drop(columns=["Packet Length"])
    
    ip_data = raw_data[["Source IP Address", "Destination IP Address"]]
    ip_data[["Int Source IP", "Int Destination IP"]] = ip_data.map(lambda x: int(IPv4Address(x)))
    ip_data[["Global Source IP","Global Destination IP"]] = ip_data[["Source IP Address", "Destination IP Address"]].map(lambda x : IPv4Address(x).is_global)
    ip_data=ip_data.drop(columns=["Source IP Address", "Destination IP Address"])

    proxy_data = raw_data["Proxy Information"].to_frame(name="Proxy Information")
    proxy_data["Is Proxy"] = proxy_data["Proxy Information"].apply(lambda x: True if pd.notna(x) and x.strip() != "" else False)
    proxy_data = proxy_data.drop(columns=["Proxy Information"])

    clean_data = raw_data.copy().drop(columns=["Timestamp", "Device Information", "Packet Length", "Source IP Address", "Destination IP Address", "Proxy Information"])
    clean_data = clean_data.join([date_data, device_data, packet_data, ip_data, proxy_data])

    list_na=clean_data.isna().any()
    list_na = list_na[list_na==True].index.tolist()
    list_na = [col for col in list_na]
    for col in list_na:
        clean_data[col] = clean_data[col].apply(lambda x: "Unknown" if pd.isna(x) else x)

    clean_data= clean_data.drop(columns=["Payload Data", "User Information"])
    clean_data.to_csv(file_path.parent.joinpath("processed_data.csv"), index=False)