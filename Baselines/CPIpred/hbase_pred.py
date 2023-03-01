# %%
import os
import pandas as pd
import numpy as np
import pathlib
import time
import logging
# %%
DATADIR = r"E:\research\cloudVM\code\data\interference_data\mul"
# %%
def find_all_csv(relative_path, file_list, file_name="-1.csv"):
    for i in os.listdir(relative_path):
        file_path = os.path.join(relative_path, i)
        if os.path.isdir(file_path):
            find_all_csv(file_path, file_list, file_name)
        else:
            if i.endswith(file_name):
                file_list.append(file_path)

# %%
csv_file_list = []
find_all_csv(DATADIR, csv_file_list, "3merge.csv")
# %%
app_csvs = {}
for i in csv_file_list:
    dirname = os.path.join(DATADIR, os.path.dirname(i))
    appname = os.path.basename(i).split('-')[0]
    if appname == 'noapp':
        continue
    if appname not in app_csvs:
        app_csvs[appname]=[]
    app_csvs[appname].append(i)
# %%
loss = {}
loss["mse"] = []
loss["mae"] = []
loss["01_precision"] = []
loss["workload"] = []
for i in app_csvs["hbase"]:
    fpath = pathlib.PurePath(i)
    workload = fpath.parent.name
    workload = workload.split("l")[-1]

    df = pd.read_csv(i)
    CPI_nostress = df[df["stress_type"] == "NO_STRESS"]["VM_CPI"].mean()
    df = df[~(df["stress_type"] == "NO_STRESS")]
    df["VM_CPI"] = df["VM_CPI"] / CPI_nostress
    lc_01 = (df["latency"] >= 1.05).astype("int")
    CPI_01 = (df["VM_CPI"] >= 1.05).astype("int")
    prec = (lc_01 == CPI_01).astype("int").sum() / len(lc_01)
    mse = ((df["VM_CPI"] - df["latency"])**2).mean()
    mae = abs(df["VM_CPI"] - df["latency"]).mean()
    loss["workload"].append(workload)
    loss["mse"].append(mse)
    loss["mae"].append(mae)
    loss["01_precision"].append(prec)
# %%
df = pd.DataFrame(loss)
df.to_csv("hbase_loss.csv", index=False)

# %%
