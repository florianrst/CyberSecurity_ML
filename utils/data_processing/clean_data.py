from executing import Source
import pandas as pd
import numpy as np
from pathlib import Path
from ipaddress import IPv4Address, IPv4Network
from sklearn.preprocessing import StandardScaler

from utils.data_processing import df_ua_parser

full_dtypes={
    "Timestamp":"string",
    "Source IP Address":"string",
    "Destination IP Address":"string",
    "Source Port":"int32",  
    "Destination Port":"int32",
    "Protocol":"category",
    "Packet Length":"int32",
    "Packet Type":"category",
    "Traffic Type":"category",
    "Payload Data":"string",
    "Malware Indicators":"category",
    "Anomaly Scores":"float64",
    "Alerts/Warnings":"category",
    "Attack Type":"category",
    "Attack Signature":"category",
    "Action Taken":"category",
    "Severity Level":"category",
    "User Information":"string",
    "Device Information":"string",
    "Network Segment":"category",
    "Geo-location Data":"string",
    "Proxy Information":"string",
    "Firewall Logs":"string",
    "IDS/IPS Alerts":"string",
    "Log Source":"string",
    "Year":"int32",
    "Month":"int32",
    "Day":"int32",
    "Hour":"int32",
    "Minute":"int32",
    "Second":"int32",
    "DayOfWeek":"int32",
    "IsWeekend":"boolean",
    "Browser Name":"string",
    "Browser Version":"string",
    "Browser Minor":"string",
    "Browser Patch":"string",
    "Browser Patch Minor":"string",
    "OS Name":"string",
    "OS Version":"string",
    "OS Version Minor":"string",
    "OS Version Patch":"string",
    "OS Version Patch Minor":"string",
    "Device Brand":"string",
    "Device Model":"string",
    "Device Type":"string",
    "Packet Bin":"string",
    "Scale Packet Length":"float64",
    "Int Source IP":"int64",
    "Int Destination IP":"int64",
    "Global Source IP":"boolean",
    "Global Destination IP":"boolean",
    "Is Proxy":"boolean"
}

raw_columns=["Timestamp", "Source IP Address", "Destination IP Address", "Source Port",
             "Destination Port", "Protocol", "Packet Length", "Packet Type", "Traffic Type",
             "Payload Data", "Malware Indicators", "Anomaly Scores", "Alerts/Warnings",
             "Attack Type", "Attack Signature", "Action Taken", "Severity Level",
             "User Information", "Device Information", "Network Segment", "Geo-location Data",
             "Proxy Information", "Firewall Logs", "IDS/IPS Alerts", "Log Source"
            ]

processed_columns = ["Source Port", "Destination Port", "Protocol", "Packet Type",
                     "Traffic Type", "Malware Indicators", "Anomaly Scores", "Alerts/Warnings",
                     "Attack Type", "Attack Signature", "Action Taken", "Severity Level",
                     "User Information", "Network Segment", "Geo-location Data",
                     "Firewall Logs", "IDS/IPS Alerts", "Log Source"
                     "Year", "Month", "Day", "Hour", "Minute", "Second", "DayOfWeek",
                     "IsWeekend","Browser Name", "Browser Version", "Browser Minor",
                     "Browser Patch", "Browser Patch Minor", "OS Name", "OS Version",
                     "OS Version Minor", "OS Version Patch", "OS Version Patch Minor",
                     "Device Brand", "Device Model", "Device Type", "Packet Bin",
                     "Scale Packet Length", "Int Source IP", "Int Destination IP",
                     "Global Source IP", "Global Destination IP", "Is Proxy"
                     ]

def load_data(file_path:Path,dtype=None):
    """Load data from a CSV file."""
    return pd.read_csv(file_path,header=0, dtype=dtype)

def transform_datetime(serie:pd.Series):
    """Split a datetime column into separate date and time components."""
    df = serie.to_frame(name="Timestamp")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df['Year'] = df["Timestamp"].dt.year
    df['Month'] = df["Timestamp"].dt.month
    df['Day'] = df["Timestamp"].dt.day
    df['Hour'] = df["Timestamp"].dt.hour
    df['Minute'] = df["Timestamp"].dt.minute
    df['Second'] = df["Timestamp"].dt.second
    df["DayOfWeek"]= df["Timestamp"].dt.dayofweek
    df["IsWeekend"]= df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
    return df.drop(columns=["Timestamp"])

def scale_packet_length(serie:pd.Series):
    df= serie.to_frame(name="Packet Length")
    scaler = StandardScaler().fit(df)
    return scaler.transform(df)

def transform_packetinfo(serie:pd.Series,scale=True):
    packet_data= serie.to_frame(name="Packet Length")
    packet_data["Packet Bin"] = packet_data["Packet Length"].apply(lambda x: "Small" if x < 420 else ("Medium" if x < 1143 else "Large"))
    if scale:
        packet_data["Scale Packet Length"] = scale_packet_length(packet_data["Packet Length"])
    return packet_data.drop(columns=["Packet Length"])

def transform_ipinfo(df:pd.DataFrame):
    columns = df.columns
    ip_columns = [f"Int {col}" for col in columns if "IP Address" in col]
    df[ip_columns] = df[columns].map(lambda x: int(IPv4Address(x)))
    global_columns = [f"Global {col}" for col in columns if "IP Address" in col]
    df[["Global Source IP","Global Destination IP"]] = df[["Source IP Address", "Destination IP Address"]].map(lambda x : IPv4Address(x).is_global)
    return df.drop(columns=columns)

def transform_proxyinfo(serie:pd.Series):
    proxy_data = serie.to_frame(name="Proxy Information")
    proxy_data["Is Proxy"] = proxy_data["Proxy Information"].apply(lambda x: True if pd.notna(x) and x.strip() != "" else False)
    return proxy_data.drop(columns=["Proxy Information"])

def clean_data(file_path, get_datetimeinfo=True, check_missing_values=True, scale_packet=True):
    df_dtypes = {col: col_type for col, col_type in full_dtypes.items() if col in raw_columns}
    raw_data = load_data(file_path,dtype=df_dtypes)
    if get_datetimeinfo:
        date_data=transform_datetime(raw_data["Timestamp"])
    else:
        date_data = pd.to_datetime(raw_data[["Timestamp"]],errors='coerce').to_frame(name="Timestamp")

    device_data = df_ua_parser(raw_data["Device Information"])
    device_data = device_data.drop(columns=["String"])

    packet_data= transform_packetinfo(raw_data["Packet Length"], scale=scale_packet)
    
    ip_data = transform_ipinfo(raw_data[["Source IP Address", "Destination IP Address"]])

    proxy_data = transform_proxyinfo(raw_data["Proxy Information"])

    clean_data = raw_data.copy().drop(columns=["Timestamp", "Device Information", "Packet Length", "Source IP Address", "Destination IP Address", "Proxy Information"])
    clean_data = clean_data.join([date_data, device_data, packet_data, ip_data, proxy_data])

    if check_missing_values:
        list_na=clean_data.isna().any()
        list_na = list_na[list_na==True].index.tolist()
        list_na = [col for col in list_na]
        for col in list_na:
            clean_data[col] = clean_data[col].apply(lambda x: "Unknown" if pd.isna(x) else x)

    clean_data= clean_data.drop(columns=["Payload Data", "User Information"])
    clean_data.to_csv(file_path.parent.joinpath("processed_data.csv"), index=False)