# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "altair==6.0.0",
#     "marimo>=0.19.6",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
#     "pandas>=3.0.0",
#     "pyarrow==23.0.0",
#     "seaborn==0.13.2",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", app_title="EDA")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##0) Importing libraries
    """)
    return


@app.cell
def _():
    import pandas as pd
    import pyarrow
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    return np, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("./data/cybersecurity_attacks.csv", encoding="utf-8")
    return (df,)


@app.cell
def _(df, mo):
    mo.ui.dataframe(df)
    return


@app.cell(hide_code=True)
def _(mo):
    # Controls
    bins_slider = mo.ui.slider(
        start=5, 
        stop=150, 
        step=5, 
        value=10, 
        label="🔍 Bins granularity"
    )
    couleurs_dict = {
        "Teal": "#008080",
        "Royal Blue": "#4169E1",
        "Indian Red": "#CD5C5C",
        "Forest Green": "#228B22",
        "Dark Orange": "#FF8C00",
        "Purple": "#9370DB",
        "Steel Blue": "#4682B4",
        "Crimson": "#DC143C",
        "Cadet Blue": "#5F9EA0",
        "Slate Gray": "#708090"
    }

    color_dropdown = mo.ui.dropdown(
        options=couleurs_dict, 
        value="Teal", 
        label="Palette chromatique"
    )

    # Sidebar
    mo.sidebar([
        mo.md("## Configuration Dashboard"),
        bins_slider,
        color_dropdown,
        mo.md("---")
    ])
    return bins_slider, color_dropdown


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##1) First look on the data
    """)
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info(memory_usage=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have 24 columns identified, 4 of them have null values : Alerts/Warnings, Proxy Information, Firewall Logs and IDS/IPSS Alerts. Moreover, this dataframe doesn't take so much space, only 7.6Mb
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##2) Analyse each column individually
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ###1-Timestamp

    Starting with the date, we noticed that is was considered as str value before. Therefore, we need to convert all the values of this column into dates.
    """)
    return


@app.cell
def _(df):
    df["Timestamp"]
    return


@app.cell
def _(df, pd):
    df["Timestamp"]=pd.to_datetime(df["Timestamp"])
    return


@app.cell
def _(df):
    df["Timestamp"]
    return


@app.cell
def _(df):
    print(df["Timestamp"].min())
    print(df["Timestamp"].max())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we see the date goes from the first january 2020 to the 11 October 2023 and isn't sorted, so we'll need to sort it in order to plot. Here is what we want to plot :
    """)
    return


@app.cell
def _(df):
    df["Timestamp"].dt.year.value_counts().sort_index()
    return


@app.cell
def _(df):
    df.groupby([df["Timestamp"].dt.year, 'Attack Type']).size().unstack(fill_value=0)
    return


@app.cell
def _(df):
    df.groupby([df["Timestamp"].dt.year, 'Attack Type']).size().unstack(fill_value=0).plot(kind="bar", stacked=True, xlabel="Year", ylabel="Count_of_year", title = "Count of the dates by year",color = ["red","green", "blue"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we assigned one specific color to each bar through the colors.
    It could be interesting to see which month cyber attacks happen, so let's plot this :
    """)
    return


@app.cell
def _(df):
    df["Timestamp"].dt.month.value_counts().sort_index()
    return


@app.cell
def _(df, np, plt):
    df["Timestamp"].dt.month.value_counts().sort_index().plot(kind="bar", xlabel="Year", ylabel="Total attacks", title = "Count of the dates by month", color = plt.cm.tab20(np.linspace(0, 1, 12)))
    return


@app.cell
def _(df, sns):
    data = df["Timestamp"].dt.month.value_counts().sort_index().reset_index()
    data.columns = ['Month', 'Count']
    sns.barplot(data=data, x='Month', y='Count', hue='Month', palette='viridis')
    return


@app.cell
def _(df, np, plt):
    df.groupby([df["Timestamp"].dt.month, 'Attack Type']).size().unstack(fill_value=0).plot(kind="bar", stacked=True, xlabel="Year", ylabel="Total attacks", title = "Count of the dates by month",color = plt.cm.tab20(np.linspace(0, 1, 12)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following is the exact equivalent
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And what about the days of the weeks on which attacks happen ?
    """)
    return


@app.cell
def _(df):
    df["Timestamp"].dt.day.value_counts().sort_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unfortunately this is the day number and not the name of the day in the week. Hopefully the datename() method allows us to get the days of the week based on the number of the days in the month.
    """)
    return


@app.cell
def _(df):
    df["Timestamp"].dt.day_name().value_counts().sort_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It's nice but it would be better to have it in the right order !
    """)
    return


@app.cell
def _():
    nice_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return (nice_order,)


@app.cell
def _(df, nice_order):
    df["Timestamp"].dt.day_name().value_counts().reindex(nice_order)
    return


@app.cell
def _(df, nice_order, np, plt):
    df["Timestamp"].dt.day_name().value_counts().reindex(nice_order).plot(kind="bar", xlabel="Year", ylabel="Count_of_attacks", title = "Count of the days", color = plt.cm.tab20(np.linspace(0, 1, 7)))
    return


@app.cell
def _(df):
    df.groupby([df["Timestamp"].dt.day_name(), 'Attack Type']).size().unstack(fill_value=0).plot(kind="bar", stacked=True, xlabel="Day_name", ylabel="Count_of_attacks", title = "Count of the dates by day",color= ["red","green", "blue"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hum... we don't have any particular trend...let's see maybe the hour of the day
    """)
    return


@app.cell
def _(df, np, plt):
    df["Timestamp"].dt.hour.value_counts().sort_index().plot(kind="bar", xlabel="Hours", ylabel="Count_of_hours", title = "Count of the hours", color = plt.cm.tab20(np.linspace(0, 1, 23)))
    return


@app.cell
def _(df):
    df.groupby([df["Timestamp"].dt.hour, 'Attack Type']).size().unstack(fill_value=0).plot(kind="bar", stacked=True, xlabel="Day_name", ylabel="Count_of_attacks", title = "Count of the dates by hour",color = ["red","green", "blue"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hum...quite similar also. We could go until the minute and the seconds...
    """)
    return


@app.cell
def _(df, np, plt):
    df["Timestamp"].dt.minute.value_counts().sort_index().plot(kind="bar", xlabel="Minutes", ylabel="Count_of_minutes", title = "Count of the minutes", color = plt.cm.tab20(np.linspace(0, 1, 59)))
    return


@app.cell
def _(df):
    df.groupby([df["Timestamp"].dt.minute, 'Attack Type']).size().unstack(fill_value=0).plot(kind="bar", stacked=True, xlabel="Minute", ylabel="Count_of_attacks", title = "Count of the dates by minutes",color = ["red","green", "blue"])
    return


@app.cell
def _(df, np, plt):
    df["Timestamp"].dt.second.value_counts().sort_index().plot(kind="bar", xlabel="Seconds", ylabel="Count_of_seconds", title = "Count of the seconds", color = plt.cm.tab20(np.linspace(0, 1, 59)))
    return


@app.cell
def _(df):
    df.groupby([df["Timestamp"].dt.second, 'Attack Type']).size().unstack(fill_value=0).plot(kind="bar", stacked=True, xlabel="Day_name", ylabel="Count_of_attacks", title = "Count of the dates by second",color = ["red","green", "blue"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nothing to see here...the data is very flat. Let's see if the time lapse between 2 attacks of the same type is homogenous.
    """)
    return


@app.cell
def _(df):
    df["User Information"].duplicated().sum()
    return


@app.cell
def _(df):
    doubles= df[df["User Information"].duplicated(keep=False)]
    doubles
    return (doubles,)


@app.cell
def _(doubles):
    doubles['Intervalle'] = doubles.sort_values(by=['User Information', 'Timestamp']).groupby('User Information')['Timestamp'].diff()
    doubles
    return


@app.cell
def _(doubles):
    doubles["Attack Type"].value_counts().plot(kind = "pie", xlabel="Attack type", ylabel = "Number of occurencies", figsize=(12, 6), title ="Count of Attacks", legend = True, autopct='%1.1f%%', textprops={'color': "black", 'weight': 'bold', 'fontsize': 14})
    return


@app.cell
def _(df):
    df_sorted = df.sort_values(by=['User Information', 'Timestamp'])
    return (df_sorted,)


@app.cell
def _(df_sorted):
    ecart_par_user_et_type = df_sorted.groupby(['User Information', 'Attack Type'])['Intervalle'].mean()
    return (ecart_par_user_et_type,)


@app.cell
def _(ecart_par_user_et_type):
    ecart_par_user_et_type.dt.total_seconds().unstack()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###2-Source IP Address

    The IP (Internet Protocol) Address uniquely identifies a network interface of a connected object to the internet (computer, smartphone, server...). It can be an IPv6 (version 6) coded on 128 bits, but on our case we see that they are coded on 32 bits (IPv4)
    """)
    return


@app.cell
def _(df):
    df["Source IP Address"]
    return


@app.cell
def _(df):
    df["Source IP Address"].value_counts() # Default sorting in descending order
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We don't have any duplicated Source IP Address. The norm RFC 1918 distinguish private and public IP address. We will analyze each IP address to see if it is public or private. According to the norm and the CIDR (Classless Inter-Domain Routing), these classes are all private IP :
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Class | Range Start | Range End | CIDR Notation | Number of Addresses |
    | :--- | :--- | :--- | :--- | :--- |
    | **Class A** | `10.0.0.0` | `10.255.255.255` | `10.0.0.0/8` | $2^{24} \approx 16.7 \text{ million}$ |
    | **Class B** | `172.16.0.0` | `172.31.255.255` | `172.16.0.0/12` | $2^{20} \approx 1.04 \text{ million}$ |
    | **Class C** | `192.168.0.0` | `192.168.255.255` | `192.168.0.0/16` | $2^{16} = 65,536$ |
    """)
    return


@app.cell
def _():
    import ipaddress as ia
    return (ia,)


@app.cell
def _(ia):
    def is_private(ip_str : str) -> bool:
        try :
            ip = ia.ip_address(ip_str)
            if ip.is_private:
                    return True
            else:
                return False
        except ValueError:
            return "Invalid IP Address"

    print(is_private("10.0.0.1"))
    print(is_private("106.109.41.24"))
    return (is_private,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our little function works and is ready to be applied to our column.
    """)
    return


@app.cell
def _(df, is_private):
    df["Source IP Address"].apply(is_private).sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have 181 source IP address that are private ! We'll store them to use them later on the study of the cyber attacks.
    """)
    return


@app.cell
def _(df, is_private):
    source_ip_is_private = df["Source IP Address"].apply(is_private)
    return (source_ip_is_private,)


@app.cell
def _(source_ip_is_private):
    source_ip_is_private
    return


@app.cell
def _(df3, source_ip_is_private):
    df3[not source_ip_is_private]["Attack Type"].value_counts().plot(kind = "pie", xlabel="Attack type", ylabel = "Number of occurencies", figsize=(12, 6), title ="Count of Attacks", legend = True, autopct='%1.1f%%', textprops={'color': "black", 'weight': 'bold', 'fontsize': 14})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's continue by separating each block of 1 byte composing the IP address to see if we find some patterns.
    """)
    return


@app.cell
def _(df):
    sip_columns = df['Source IP Address'].str.split('.', expand=True).astype(int)
    print(sip_columns)
    return (sip_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The option expand = True splits in the different columns. We'll need to rename our columns.
    """)
    return


@app.cell
def _(sip_columns):
    sip_columns.columns = ['SIP_O1', 'SIP_O2', 'SIP_O3', 'SIP_O4']
    print(sip_columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And then we add it to our original dataframe to have all the data inside a single table
    """)
    return


@app.cell
def _(df, pd, sip_columns):
    df2 = pd.concat([df, sip_columns], axis=1)
    print(df2)
    return (df2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have all our 4 new columns, let's plot them individually.
    """)
    return


@app.cell
def _(df2):
    df2["SIP_O1"].describe()
    return


@app.cell
def _(bins_slider, color_dropdown, df2):
    df2["SIP_O1"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of 1st part of the source IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 1st part of the Source IP address")
    return


@app.cell
def _(df2):
    df2.pivot(columns='Attack Type', values='SIP_O1')
    return


@app.cell
def _(bins_slider, df2):
    df2.pivot(columns='Attack Type', values='SIP_O1').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of 1st part of the source IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 1st part of the Source IP address")
    return


@app.cell
def _(df2):
    df2["SIP_O2"].describe()
    return


@app.cell
def _(bins_slider, color_dropdown, df2):
    df2["SIP_O2"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of 2nd part of the source IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 2nd part of the Source IP address")
    return


@app.cell
def _(bins_slider, df2):
    df2.pivot(columns='Attack Type', values='SIP_O2').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of 2nd part of the source IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 2nd part of the Source IP address")
    return


@app.cell
def _(df2):
    df2["SIP_O3"].describe()
    return


@app.cell
def _(bins_slider, color_dropdown, df2):
    df2["SIP_O3"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of 3rd part of the source IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 3rd part of the Source IP address")
    return


@app.cell
def _(bins_slider, df2):
    df2.pivot(columns='Attack Type', values='SIP_O3').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of 3rd part of the source IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 3rd part of the Source IP address")
    return


@app.cell
def _(df2):
    df2["SIP_O4"].describe()
    return


@app.cell
def _(bins_slider, color_dropdown, df2):
    df2["SIP_O4"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of 4th part of the source IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 4th part of the Source IP address")
    return


@app.cell
def _(bins_slider, df2):
    df2.pivot(columns='Attack Type', values='SIP_O4').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of 4th part of the source IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 4th part of the Source IP address")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similarly we don't have any trend. We'll do exactly the same things with the Destination IP Address...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###3-Destination IP Address
    """)
    return


@app.cell
def _(df):
    df["Destination IP Address"]
    return


@app.cell
def _(df):
    df["Destination IP Address"].value_counts() # Default sorting in descending order
    return


@app.cell
def _(df, is_private):
    df["Destination IP Address"].apply(is_private).sum()
    return


@app.cell
def _(df, is_private):
    destination_ip_is_private = df["Destination IP Address"].apply(is_private)
    return


@app.cell
def _(df):
    dip_columns = df['Destination IP Address'].str.split('.', expand=True).astype(int)
    print(dip_columns)
    return (dip_columns,)


@app.cell
def _(dip_columns):
    dip_columns.columns = ['DIP_O1', 'DIP_O2', 'DIP_O3', 'DIP_O4']
    print(dip_columns)
    return


@app.cell
def _(df2, dip_columns, pd):
    df3 = pd.concat([df2, dip_columns], axis=1)
    print(df3)
    return (df3,)


@app.cell
def _(df3):
    df3["DIP_O1"].describe()
    return


@app.cell
def _(bins_slider, color_dropdown, df3):
    df3["DIP_O1"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of 1st part of the destination IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 1st part of the Destination IP address")
    return


@app.cell
def _(bins_slider, df3):
    df3.pivot(columns='Attack Type', values='DIP_O1').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of 1st part of the Destination IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 1st part of the Destination IP address")
    return


@app.cell
def _(df3):
    df3["DIP_O2"].describe()
    return


@app.cell
def _(bins_slider, color_dropdown, df3):
    df3["DIP_O2"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of 2nd part of the destination IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 2nd part of the Destination IP address")
    return


@app.cell
def _(bins_slider, df3):
    df3.pivot(columns='Attack Type', values='DIP_O2').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of 2nd part of the Destination IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 2nd part of the Destination IP address")
    return


@app.cell
def _(df3):
    df3["DIP_O3"].describe()
    return


@app.cell
def _(bins_slider, color_dropdown, df3):
    df3["DIP_O3"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of 3rd part of the destination IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 3rd part of the Destination IP address")
    return


@app.cell
def _(bins_slider, df3):
    df3.pivot(columns='Attack Type', values='DIP_O3').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of 3rd part of the Destination IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 3rd part of the Destination IP address")
    return


@app.cell
def _(df3):
    df3["DIP_O4"].describe()
    return


@app.cell
def _(bins_slider, color_dropdown, df3):
    df3["DIP_O4"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of 4th part of the destination IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 4th part of the Destination IP address")
    return


@app.cell
def _(bins_slider, df3):
    df3.pivot(columns='Attack Type', values='DIP_O4').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of 4th part of the Destination IP address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 4th part of the Destination IP address")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As before, we can't make any conclusions on the Destionation IP Address.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###4-Source Port
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In technical cybersecurity analysis, the Source Port is a 16-bit integer (ranging from 0 to 65535 : 2^16-1) used by the client to uniquely identify a communication session. While the Destination Port tells you which service is being targeted (e.g., 443 for HTTPS), the Source Port provides a "behavioral fingerprint" of the attacker's operating system or tools.
    """)
    return


@app.cell
def _(df3):
    df3["Source Port"]
    return


@app.cell
def _(mo):
    mo.md(r"""
    Legitimate client applications (browsers, apps) almost never use a source port below 1,024. This is our case : we don't have any values under this threshold.
    """)
    return


@app.cell
def _(df3):
    df3["Source Port"].describe()
    return


@app.cell
def _(df3):
    df3["Source Port"].value_counts().sort_index()
    return


@app.cell
def _(mo):
    mo.md(r"""
    To analyze ports effectively, you must distinguish between the three official categories defined by IANA (Internet Assigned Numbers Authority):
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | Category | Port Range | Usage Description |
    | :--- | :--- | :--- |
    | **Well-Known Ports** | `0 – 1,023` | Reserved for system/administrative services (SSH, HTTP, etc.). |
    | **Registered Ports** | `1,024 – 49,151` | Assigned to specific user processes or applications. |
    | **Dynamic/Ephemeral Ports** | `49,152 – 65,535` | Temporary ports used by clients for outgoing connections. |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's write a little function to use this classification and apply it to our source port
    """)
    return


@app.function
def is_registered(p : int) -> bool:
    if p < 49152: 
        return True
    return False


@app.cell
def _(df3):
    df3["Source Port"].apply(is_registered)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similarly, we'll store this boolean mask for a reuse.
    """)
    return


@app.cell
def _(df3):
    is_sourceport_registered = df3["Source Port"].apply(is_registered)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now we can plot the values of the Source port in a histogramm showing the number of values for each categories
    """)
    return


@app.cell
def _(bins_slider, color_dropdown, df3):
    df3["Source Port"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of Source Port", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the Source Port")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###5-Destination Port
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll do the same for the destination port.
    """)
    return


@app.cell
def _(df3):
    df3["Destination Port"]
    return


@app.cell
def _(df3):
    df3["Destination Port"].describe()
    return


@app.cell
def _(df3):
    df3["Destination Port"].value_counts().sort_index()
    return


@app.cell
def _(df3):
    df3["Destination Port"].apply(is_registered)
    return


@app.cell
def _(bins_slider, color_dropdown, df3):
    df3["Destination Port"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of Destination Port", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the Destination Port")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###6-Protocol
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Data is transferred from hardware to another in packets. But these packets are different according to the network protocol used. We distinguish 3 main categories of protocol : ICMP (Internet Control Message Protocol), UDP (User Datagram Protocol), TCP (Transmission Control Protocol). Each protocol handles data packets with a distinct philosophy regarding reliability, speed, and overhead. Let's have a look on this table :

    | Protocol | Full Name | Connection Type | Reliability & Delivery | Typical Header Size | Key Features | Common Use Cases |
    |---------|----------|----------------|------------------------|--------------------|--------------|------------------|
    | TCP | Transmission Control Protocol | Connection-oriented | Guaranteed: Uses acknowledgments (ACK) and retransmission of lost segments. | 20 – 60 Bytes | 3-way handshake (SYN, SYN-ACK, ACK), sequence numbers, flow control. | HTTPS, SSH, FTP, Database queries |
    | UDP | User Datagram Protocol | Connectionless | Best-effort: No guarantee of delivery, order, or error recovery. | 8 Bytes | Minimal overhead, stateless, no handshake, low latency. | VoIP, DNS, Video streaming, Online gaming |
    | ICMP | Internet Control Message Protocol | Connectionless | Diagnostic: Used for error reporting and network signaling; does not carry user data. | Variable | Operates at the Network Layer (IP); uses "Type" and "Code" fields instead of ports. | Ping (Echo), Traceroute, Network congestion alerts |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We could use some analogies to remember these :
    TCP is like a phone call, where both sides establish a connection and every message is acknowledged;
    UDP is like shouting information across a room, fast but with no guarantee it’s heard or in order;
    ICMP is like road signs and warning lights, not used to carry messages, but to report problems and network status. But in which cases to use what in cyber attacks ?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | Protocol | Common Attack Scenarios | Common Ports | Why These Ports Are Used |
    |--------|------------------------|-------------|--------------------------|
    | **TCP** | Data exfiltration, Command & Control (C2), brute-force attacks (SSH/RDP), web attacks (SQL injection, credential stuffing) | 80 (HTTP), 443 (HTTPS), 22 (SSH), 21 (FTP), 3389 (RDP) | These ports are usually **open and trusted**, especially 80/443 which blend into normal web traffic; 443 adds **encryption**, making inspection difficult; 22 and 3389 provide **direct remote access** to systems. |
    | **UDP** | DDoS amplification, UDP flood, fast network scanning | 53 (DNS), 123 (NTP), 1900 (SSDP) | UDP allows **IP spoofing** and has **no handshake**, enabling massive traffic; these services respond with **larger replies**, amplifying attack volume and overwhelming targets. |
    | **ICMP** | Network reconnaissance (ping sweep), path mapping (traceroute), ICMP flood, Smurf attacks | N/A (uses Type & Code, not ports) | ICMP is often **allowed by default**, lightly monitored, and reveals **network reachability and structure**, making it useful for discovery and disruption rather than data transfer. |
    """)
    return


@app.cell
def _(mo):
    mo.image("public/Network_protocols.png", width = 1000)
    return


@app.cell
def _(df3):
    df3["Protocol"]
    return


@app.cell
def _(df3):
    df3["Protocol"].value_counts().plot(kind = "pie", xlabel="Type of Protocol", ylabel = "Number of occurencies", figsize=(12, 6), title ="Count of Protocols", legend = True, autopct='%1.1f%%', textprops={'color': "black", 'weight': 'bold', 'fontsize': 14})
    return


@app.cell
def _(mo):
    mo.md(r"""
    Similarly we don't have any trend, but rather an equal proportion of each Protocol.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###7-Packet Length
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We saw that the length of the packet varies between the protocol.
    """)
    return


@app.cell
def _(df3):
    df3["Packet Length"]
    return


@app.cell
def _(df3):
    df3["Packet Length"].describe()
    return


@app.cell
def _(df3):
    df3["Packet Length"].value_counts().sort_index()
    return


@app.cell
def _(bins_slider, color_dropdown, df3):
    df3["Packet Length"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of the Packet Length", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the Packet Length")
    return


@app.cell
def _(mo):
    mo.md(r"""
    No trend visible.
    """)
    return


@app.cell
def _(mo):
    split_slider = mo.ui.slider(start=64, stop=1500, step=50, value=500, label="Cut threshold (Octets)")
    split_slider
    return (split_slider,)


@app.cell
def _(df3, plt, split_slider):
    color_map = {
        "TCP": "#4169E1",   # Royal Blue
        "UDP": "#FF8C00",   # Dark Orange
        "ICMP": "#228B22"   # Forest Green
    }

    # Préparation des données : on crée une colonne par protocole
    pivot_df = df3.pivot(columns="Protocol", values="Packet Length").fillna(0)
    print(pivot_df)
    # Tracé de l'histogramme empilé
    ax1 = pivot_df.plot(
        kind="hist",
        bins=[64, split_slider.value, 1500],
        stacked=True, # Empile les barres les unes sur les autres
        color=[color_map[c] for c in pivot_df.columns if c in color_map],
        edgecolor="black",
        figsize=(12, 6)
    )

    ax1.set_title("Total volume for each segment and protocol", color="white")
    ax1.legend(facecolor='#1e1e1e', labelcolor='white')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    For any split point, the data is flat.
    """)
    return


@app.cell
def _(df3):
    # 1. Préparation des données par protocole
    protocols = ["TCP", "UDP", "ICMP"]
    data_to_plot = [df3[df3["Protocol"] == p]["Packet Length"] for p in protocols]
    data_to_plot
    return data_to_plot, protocols


@app.cell
def _(color_dropdown, data_to_plot, plt, protocols):
    # 2. Création du graphique
    fig, ax = plt.subplots(figsize=(12, 7))

    # 3. Création du Violin Plot
    parts = ax.violinplot(data_to_plot, showmedians=True)

    # 4. Stylisation avec vos paramètres Marimo
    for pc in parts['bodies']:
        pc.set_facecolor(color_dropdown.value) # Utilise votre sélection
        pc.set_edgecolor('white')
        pc.set_alpha(0.7)

    # Colorer les lignes statistiques
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor("white")
        vp.set_linewidth(1)

    # 5. Cosmétique "Dark Mode"
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(protocols, color="white", fontsize=12)
    ax.set_title("Packet Length for each Protocol", color="white", fontsize=14)
    ax.set_ylabel("Packet size", color="white")
    ax.tick_params(colors='white')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###8-Packet Type
    """)
    return


@app.cell
def _(df3):
    df3["Packet Type"]
    return


@app.cell
def _(df3):
    df3["Packet Type"].value_counts()
    return


@app.cell
def _(df3, pd):
    pd.crosstab(df3["Packet Type"], df3["Protocol"],values =df3["Packet Length"], aggfunc = "mean")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###9-Traffic Type
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(df3):
    df3["Traffic Type"]
    return


@app.cell
def _(df3):
    df3["Traffic Type"].value_counts()
    return


@app.cell
def _(df3):
    df3["Traffic Type"].value_counts().plot(kind = "pie", xlabel="Type of Traffic", ylabel = "Number of occurencies", figsize=(12, 6), title ="Count of Traffic", legend = True, autopct='%1.1f%%', textprops={'color': "black", 'weight': 'bold', 'fontsize': 14})
    return


@app.cell
def _(df3, pd):
    pd.crosstab(df3["Traffic Type"], df3["Protocol"],values =df3["Packet Length"], aggfunc = "count")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###10-Payload data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    There is nothing to see here, the payload data is randomly generated text.
    """)
    return


@app.cell
def _(df):
    df["Payload Data"].value_counts()
    return


@app.cell
def _():
    ###11-Malware Indicators
    return


@app.cell
def _(mo):
    mo.md(r"""
    The indicator of compromise (IoC) is crucial as it indicates a computer intrusion with high confidence
    """)
    return


@app.cell
def _(df):
    df["Malware Indicators"]
    return


@app.cell
def _(df):
    df["Malware Indicators"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We do have 20 000 (the half of the dataset) of missing values. Let's see how is the data with other categories.
    """)
    return


@app.cell
def _(df3, pd):
    pd.crosstab(df3["Traffic Type"], df3["Protocol"],df3["Malware Indicators"], aggfunc="count")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll make a mask to filter and use it afterwards.
    """)
    return


@app.cell
def _(df3):
    malware_indicator = df3["Malware Indicators"]=="IoC Detected"
    return (malware_indicator,)


@app.cell
def _(df3, malware_indicator):
    df3[malware_indicator]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###12-Anomaly Scores
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The anomaly score is a percentage given on each attack.
    """)
    return


@app.cell
def _(df3):
    df3["Anomaly Scores"]
    return


@app.cell
def _(df3):
    df3["Anomaly Scores"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###13-Alerts/Warnings
    """)
    return


@app.cell
def _(df3):
    df3["Alerts/Warnings"]
    return


@app.cell
def _(df3):
    df3["Alerts/Warnings"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###14-Attack type
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This is our column to predict. We need to understand which type of attack it represents. First, it is essential to say that cyber attacks can be completely different, we could have all of them :
    - Clickjacking
    - Phishing
    - Identity Theft
    - Credential Stuffing
    - DDoS Attack
    - Brute Force
    - Eavesdropping
    - MITM (Man-in-the-middle)
    - Typosquatting
    - Insider Threat
    - Social Engineering
    - SQL Injection
    - DNS Poisoning
    - Drive-by Download
    - XSS (Cross-site scripting)
    - IoT Exploitation
    - Zero Day Exploit
    - Supply Chain Attack
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    But fortunately, we'll only study three of these :

    | Attack Type | Description | Main Goal | Typical Methods / Vectors |
    |------------|-------------|-----------|----------------------------|
    | **DDoS (Distributed Denial of Service)** | Overwhelms a system, server, or network with massive traffic from many sources, making it unavailable to legitimate users. | Disrupt availability | Traffic flooding, amplification attacks (DNS, NTP, SSDP), botnets |
    | **Malware** | Malicious software designed to damage, disrupt, spy on, or gain unauthorized access to systems. | Compromise systems and data | Trojans, ransomware, spyware, worms, infected downloads |
    | **Intrusion** | Unauthorized access to a system or network by exploiting vulnerabilities or weak credentials. | Gain control or steal information | Exploiting vulnerabilities, stolen credentials, misconfigurations |
    """)
    return


@app.cell
def _(df3):
    df3["Attack Type"]
    return


@app.cell
def _(df3):
    df3["Attack Type"].value_counts()
    return


@app.cell
def _(df3):
    df3["Attack Type"].value_counts().plot(kind = "pie", xlabel="Attack type", ylabel = "Number of occurencies", figsize=(12, 6), title ="Count of Attacks", legend = True, autopct='%1.1f%%', textprops={'color': "black", 'weight': 'bold', 'fontsize': 14})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As it is our target variable, we'll make a mask on each type of attacks to study further in detail the repartition of each.
    """)
    return


@app.cell
def _(df3):
    ddos = df3["Attack Type"] == "DDoS"
    return


@app.cell
def _(df3):
    malware = df3["Attack Type"] == "Malware"
    return


@app.cell
def _(df3):
    intrusion = df3["Attack Type"] == "Intrusion"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###15-Attack signature
    """)
    return


@app.cell
def _(df3):
    df3["Attack Signature"]
    return


@app.cell
def _(df3):
    df3["Attack Signature"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###16-Action taken
    """)
    return


@app.cell
def _(df3):
    df3["Action Taken"].value_counts()
    return


@app.cell
def _(color_dropdown, df3):
    df3["Action Taken"].value_counts().sort_index().plot(kind="bar", color=color_dropdown.value, xlabel="Type of Action taken", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Action Taken")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###17-Severity Level
    """)
    return


@app.cell
def _(df3):
    df3["Severity Level"].value_counts()
    return


@app.cell
def _(color_dropdown, df3):
    df3["Severity Level"].value_counts().plot(kind = "bar", color = color_dropdown.value, xlabel="Severity Level", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Severity Level")
    return


@app.cell
def _(df3, pd):
    pd.crosstab(df3["Severity Level"], [df3["Attack Type"], df3["Action Taken"]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###18-User Information
    """)
    return


@app.cell
def _(df3):
    df3["User Information"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Many users seem to have been targeted by attacks many times.
    """)
    return


@app.cell
def _(df3):
    # Conversion en DataFrame avec renommage de la colonne calculée
    resultat = df3.groupby(["User Information", "Attack Type"])["Attack Type"].count().reset_index(name='Occurrence')
    resultat
    return


@app.cell
def _(df3):
    # Transformation du MultiIndex en matrice pivotée
    df_pivot = df3.groupby(["User Information", "Attack Type"])["Attack Type"].count().unstack(fill_value=0)
    df_pivot
    return


@app.cell
def _(color_dropdown, df3):
    # 1. Regroupement par les deux dimensions
    # On utilise to_frame() pour transformer la Series résultante en un DataFrame propre
    df_final = df3.groupby(["User Information", "Attack Type"])["Attack Type"].count().to_frame(name="Total Number").sort_values(by="Total Number", ascending=False)

    # 2. Affichage
    df_critique = df_final[df_final["Total Number"] >=3]
    distribution = df_critique.groupby(level="Attack Type")["Total Number"].sum()
    distribution.plot(kind = "bar", color = color_dropdown.value, xlabel="Attack Type", ylabel = "Number of occurencies for users targeted 3 times minimum", figsize=(12, 6), edgecolor="black", title ="User targeted multiples times '3x minimum'")
    return (df_final,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    User targeted minimum 3 times seems to have face more intrusions.
    """)
    return


@app.cell
def _(df_final):
    df_final
    return


@app.cell
def _(color_dropdown, df3, pd):
    diversite_attaques = df3.groupby("User Information")["Attack Type"].nunique()
    same_type_total = (diversite_attaques == 1).sum()
    diff_types_total = (diversite_attaques >= 2).sum()
    df_counts = pd.DataFrame({
        'Category': ['Same attacks', 'Different attacks'],
        'Number of attacks': [same_type_total, diff_types_total]
    })
    df_counts.plot(kind = "bar", x="Category", y = "Number of attacks", color = color_dropdown.value, xlabel="Same or different attacks", ylabel = "Number of same and differents attacks respectively", figsize=(12, 6), edgecolor="black", title ="User targeted by different types of attacks")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###19-Device Information
    """)
    return


@app.cell
def _(df3):
    df3["Device Information"]
    return


@app.cell
def _(mo):
    mo.md(r"""
    This column contains 6 different types of information that we'll use that are all part of the "User Agent", which is  included in every HTTP header in requests to identify the client :
    - The first part, like "Mozilla", is the client's webbrowser
    - The second part, like "Macintosh", is the client's OS
    - The third part is the layout engine, used to render the HTML, CSS, Javascript, for instance Trident (Internet Explorer), Gecko (Firefox) or Presto (Opera)
    - The fourth part is the web browser specific version, like "MSIE 9.0"
    - The fifth part is the architecture, the type of processor, like PPC (Power PC) or i686 (Intel 32 bits)
    - The last part is the language and region, like "nl" for Netherlands

    The majority of cyber attacks usually don't target the browser directly, which is the visible part (the UI), but use OS or layout engine vulnerabilities (...)
    To better see each variable, we'll use the python user agent library to split our column into the most useful parts.
    """)
    return


@app.cell
def _(df3, pd):
    from ua_parser import parse

    # Supposons que 'df' est votre DataFrame et 'user_agent_column' votre colonne
    def process_ua(ua_string):
        ua = parse(ua_string)
        return pd.Series({
            'OS_Family': ua.os.family,
            'OS_Version': ua.os.version_string,
            'Browser': ua.browser.family,
            'Browser_Version': ua.browser.version_string,
            'Device_Type': 'Mobile' if ua.is_mobile else ('Tablet' if ua.is_tablet else 'Desktop'),
            'Is_Bot': ua.is_bot
        })

    # Application de la fonction sur le DataFrame
    df_info = df3['Device Information'].apply(process_ua)
    df_info

    return (df_info,)


@app.cell
def _(mo):
    mo.md(r"""
    #### Analysis of the OS Family and OS Version
    """)
    return


@app.cell
def _(df_info):
    df_info["OS_Family"].value_counts()
    return


@app.cell
def _(color_dropdown, df_info):
    df_info["OS_Family"].value_counts().plot(kind = "bar", x="OS_Family", y = "count", color = color_dropdown.value, xlabel="OS_Family", ylabel = "Number", figsize=(12, 6), edgecolor="black", title ="Repartition of OS")
    return


@app.cell
def _(mo):
    mo.md(r"""
    We identify a majority of Windows User. But what is the repartition in term of attacks inside these categories ?
    """)
    return


@app.cell
def _(df3, df_info, pd):
    pd.crosstab(df_info['OS_Family'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks by OS Family",ylabel="Number of attacks",xlabel="OS Family")

    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's analyze in detail the OS families.
    """)
    return


@app.cell
def _(df_info):
    windows=df_info[df_info['OS_Family']=='Windows']
    return (windows,)


@app.cell
def _(df3, pd, windows):
    pd.crosstab(windows['OS_Version'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for Windows by OS Version",ylabel="Number of attacks",xlabel="OS Version for Windows")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Linux Users don't have specific OS Versions so we pass them.
    """)
    return


@app.cell
def _(df_info):
    ios=df_info[df_info['OS_Family']=='iOS']
    ios
    return (ios,)


@app.cell
def _(df3, ios, pd):
    pd.crosstab(ios['OS_Version'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for IOS by OS Version",ylabel="Number of attacks",xlabel="OS Version for IOS")
    return


@app.cell
def _(df_info):
    macos=df_info[df_info['OS_Family']=='Mac OS X']
    macos
    return (macos,)


@app.cell
def _(df3, macos, pd):
    pd.crosstab(macos['OS_Version'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for MAC OS X by OS Version",ylabel="Number of attacks",xlabel="OS Version for MAC OS X")
    return


@app.cell
def _(df_info):
    android=df_info[df_info['OS_Family']=='Android']
    return (android,)


@app.cell
def _(android, df3, pd):
    pd.crosstab(android['OS_Version'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for Android by OS Version",ylabel="Number of attacks",xlabel="OS Version for Android")
    return


@app.cell
def _(mo):
    mo.md(r"""
    We could analyze in detail each vulnerability with the National Vulnerability Database
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Analysis of the Browser and browser version
    """)
    return


@app.cell
def _(df_info):
    df_info["Browser"].value_counts()
    return


@app.cell
def _(color_dropdown, df_info):
    df_info["Browser"].value_counts().plot(kind = "bar", x="Browser", y = "count", color = color_dropdown.value, xlabel="Browsers", ylabel = "Number", figsize=(12, 6), edgecolor="black", title ="Repartition of Browsers")
    return


@app.cell
def _(df3, df_info, pd):
    pd.crosstab(df_info['Browser'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks by Browser",ylabel="Number of attacks",xlabel="Browser")
    return


@app.cell
def _(df_info):
    chrome=df_info[df_info['Browser']=='Chrome']
    chrome
    return (chrome,)


@app.cell
def _(chrome, df3, pd):
    pd.crosstab(chrome['Browser_Version'], df3['Attack Type'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We take only the main version of chrome because there are too much versions.
    """)
    return


@app.cell
def _(chrome, df3, pd):
    pd.crosstab(chrome['Browser_Version'].str[0:2], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for Chrome by Browser Version",ylabel="Number of attacks",xlabel="Browser Version for Chrome")
    return


@app.cell
def _(df_info):
    opera=df_info[df_info['Browser']=='Opera']
    opera
    return (opera,)


@app.cell
def _(df3, opera, pd):
    pd.crosstab(opera['Browser_Version'].str.split(".").str[0], df3['Attack Type'])
    return


@app.cell
def _(df3, opera, pd):
    pd.crosstab(opera['Browser_Version'].str.split(".").str[0], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for Opera by Browser Version",ylabel="Number of attacks",xlabel="Browser Version for Opera")
    return


@app.cell
def _(df_info):
    opera_v8=df_info[(df_info['Browser']=='Opera') & (df_info['Browser_Version'].str.split(".").str[0]=='8')]
    opera_v8
    return (opera_v8,)


@app.cell
def _(df3, opera_v8, pd):
    pd.crosstab(opera_v8['Browser_Version'].str.split(".").str[1], df3['Attack Type'])
    return


@app.cell
def _(df3, opera_v8, pd):
    pd.crosstab(opera_v8['Browser_Version'].str.split(".").str[1], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for Opera by Browser Version",ylabel="Number of attacks",xlabel="Browser Version for Opera")
    return


@app.cell
def _(df_info):
    ie=df_info[df_info['Browser']=='IE']
    ie
    return (ie,)


@app.cell
def _(df3, ie, pd):
    pd.crosstab(ie['Browser_Version'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for IE by Browser Version",ylabel="Number of attacks",xlabel="Browser Version for IE")
    return


@app.cell
def _(df_info):
    safari=df_info[df_info['Browser']=='Safari']
    safari
    return (safari,)


@app.cell
def _(df3, pd, safari):
    pd.crosstab(safari['Browser_Version'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for Safari by Browser Version",ylabel="Number of attacks",xlabel="Browser Version for Safari")
    return


@app.cell
def _(df_info):
    firefox=df_info[df_info['Browser']=='Firefox']
    firefox
    return (firefox,)


@app.cell
def _(df3, firefox, pd):
    pd.crosstab(firefox['Browser_Version'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for Safari by Browser Version",ylabel="Number of attacks",xlabel="Browser Version for Safari")
    return


@app.cell
def _(df_info):
    mobile_safari=df_info[df_info['Browser']=='Mobile Safari']
    mobile_safari
    return (mobile_safari,)


@app.cell
def _(df3, mobile_safari, pd):
    pd.crosstab(mobile_safari['Browser_Version'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks for Mobile Safari by Browser Version",ylabel="Number of attacks",xlabel="Browser Version for Mobile Safari")
    return


@app.cell
def _(mo):
    mo.md(r"""
    We don't have any tendency visible...
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Analysis of the Device Type
    """)
    return


@app.cell
def _(df_info):
    df_info["Device_Type"].value_counts()
    return


@app.cell
def _(df3, df_info, pd):
    pd.crosstab(df_info["Device_Type"], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks by Device Type",ylabel="Number of attacks",xlabel="Device Type")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###20-Network Segment
    """)
    return


@app.cell
def _(df3):
    df3["Network Segment"].value_counts()
    return


@app.cell
def _(df3, pd):
    pd.crosstab(df3['Network Segment'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks by Network Segment",ylabel="Number of attacks",xlabel="Network Segment")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###21-Geo-location data
    """)
    return


@app.cell
def _(df4):
    df4["Geo-location Data"]
    return


@app.cell
def _(df):
    df['City'] = df['Geo-location Data'].str.split(',').str[0].str.strip()
    df['State'] = df['Geo-location Data'].str.split(',').str[1].str.strip()
    return


@app.cell
def _(df):
    df['City']
    return


@app.cell
def _(geolocator):
    location_1 = geolocator.geocode("Jamshedpur, India", timeout=10)
    print(location_1.longitude, location_1.latitude)
    return


@app.cell
def _(df):
    villes_uniques_2 = df['City'].unique()
    len(villes_uniques_2)
    return


@app.cell
def _(df, pd):
    villes_uniques = df['Geo-location Data'].unique()
    cache_geo = {}
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    import time

    # Augmentation du timeout à 10 secondes pour éviter les ReadTimeoutError
    geolocator = Nominatim(user_agent="cyber_security_analysis_project")

    print(f"Début du géocodage pour {len(villes_uniques)} localisations uniques...")

    for loc in villes_uniques:
        try:
            # Ajout de ", India" pour limiter le périmètre de recherche
            location = geolocator.geocode(f"{loc}, India", timeout=10)
            if location:
                cache_geo[loc] = (location.latitude, location.longitude)
        
            # Pause obligatoire de 1.5s pour respecter les serveurs gratuits
            time.sleep(1) 
        
        except Exception as e:
            print(f"Erreur sur {loc}: {e}")
            continue

    # 3. Application du cache au DataFrame principal
    df['coords'] = df['Geo-location Data'].map(cache_geo)
    df[['lat', 'lon']] = pd.DataFrame(df['coords'].tolist(), index=df.index)
    return (geolocator,)


@app.cell
def _(df):
    df['coords']
    return


@app.cell
def _(df_geo):
    import folium

    # Dictionnaire de correspondance pour vos 3 couleurs
    colors = {
        'DDoS': 'red',
        'Malware': 'blue',
        'Intrusion': 'green'
    }

    # Initialisation de la carte centrée sur l'Inde
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    # Exemple d'itération sur vos données
    # Supposons un DataFrame 'df_geo' avec les colonnes 'lat', 'lon' et 'Attack Type'
    for index, row in df_geo.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            color=colors.get(row['Attack Type'], 'gray'),
            fill=True,
            fill_opacity=0.7,
            popup=f"Type: {row['Attack Type']}\nLoc: {row['geolocation']}"
        ).add_to(m)

    m.save('cyber_map.html')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###22-Proxy Information
    """)
    return


@app.cell
def _(df3):
    df3["Proxy Information"]
    return


@app.cell
def _(mo):
    mo.md(r"""
    The proxy columns has 19851 null values. Why ? Let's see if it is homogenous for the attack type
    """)
    return


@app.cell
def _(df3, pd):
    pd.crosstab(df3["Proxy Information"].isna(), df3["Attack Type"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Yes it is perfectly homogenous...
    """)
    return


@app.cell
def _(df3):
    df3.loc[:,["Source IP Address", "Attack Type","Destination IP Address", "Proxy Information"]]
    return


@app.cell
def _(mo):
    mo.md(r"""
    We split the Proxy Info in 4 blocks exactly as we did for the Source and Destination IP Address
    """)
    return


@app.cell
def _(df):
    proxy_columns = df['Proxy Information'].dropna().str.split('.', expand=True).astype(int)
    print(proxy_columns)
    return (proxy_columns,)


@app.cell
def _(proxy_columns):
    proxy_columns.columns = ['Proxy_O1', 'Proxy_O2', 'Proxy_O3', 'Proxy_O4']
    print(proxy_columns)
    return


@app.cell
def _(df3, pd, proxy_columns):
    df4 = pd.concat([df3, proxy_columns], axis=1)

    return (df4,)


@app.cell
def _(df4):
    print(df4.iloc[:,25:])
    return


@app.cell
def _(bins_slider, color_dropdown, df4):
    df4["Proxy_O1"].plot(kind="hist", bins=bins_slider.value, color=color_dropdown.value, xlabel="Values of 1st part of the Proxy address", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 1st part of the Proxy address")
    return


@app.cell
def _(bins_slider, df4):
    df4.pivot(columns='Attack Type', values='Proxy_O1').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of the 1st part of the Proxy Aderess", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 1st part of the Proxy Address")
    return


@app.cell
def _(bins_slider, df4):
    df4.pivot(columns='Attack Type', values='Proxy_O2').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of the 2nd part of the Proxy Aderess", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 2nd part of the Proxy Address")
    return


@app.cell
def _(bins_slider, df4):
    df4.pivot(columns='Attack Type', values='Proxy_O3').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of the 3rd part of the Proxy Aderess", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 3rd part of the Proxy Address")
    return


@app.cell
def _(bins_slider, df4):
    df4.pivot(columns='Attack Type', values='Proxy_O4').plot(kind="hist",stacked=True, bins=bins_slider.value, colormap='tab10', xlabel="Values of the 4th part of the Proxy Aderess", ylabel = "Number of occurencies", figsize=(12, 6), edgecolor="black", title ="Count of the 4th part of the Proxy Address")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###23-Firewall Logs
    """)
    return


@app.cell
def _(df4):
    df4["Firewall Logs"]
    return


@app.cell
def _(df3, df4, pd):
    pd.crosstab(df4['Firewall Logs'], df3['Attack Type']).plot(kind="bar", stacked=True,figsize=(12, 6),edgecolor="black",title="Repartition of attacks by Firewall Logs",ylabel="Number of attacks",xlabel="Firewall Logs")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###24-IDS/IPS Alerts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###25-Log source
    """)
    return


if __name__ == "__main__":
    app.run()
