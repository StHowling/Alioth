# %%
import os
import pandas as pd
import numpy as np
import pathlib
from sklearn.mixture import GaussianMixture
from torch import threshold
# %%
DATADIR = r"data/practical"
# DATADIR = r"E:\research\cloudVM\code\data\interference_data\mul"
wi = {
    "cassandra": 5,
    "etcd": 5,
    "hbase": 5,
    "kafka": 3,
    "milc": 2,
    "mongoDB": 5,
    "rabbitmq": 4,
    "redis": 5
}
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
find_all_csv(DATADIR, csv_file_list, ".csv")
# %%
result = {}
result["app"] = []
result["01_acc"] = []
result["mse"] = []
result["mae"] = []
result["threshold"] = []
# thre = [1.05]
for i in csv_file_list:
    filename = os.path.basename(i)
    app = filename.split(".")[0]
    print(app)
    if app == "total":
        continue
    df = pd.read_csv(i)
    X = df["VM_CPI"].values.reshape(-1, 1)
    gm = GaussianMixture(n_components=10, random_state=0).fit(X)
    CPI_nostress = gm.means_.min()
    prec = 0
    mse = 0
    mae = 0
    cpi = df["VM_CPI"] / CPI_nostress
    qos = df["QoS"]
    for j in np.arange(1.05, 1.3, 0.05):
        CPI_01 = (cpi >= j).astype("int")
        QoS_01 = (qos >= j).astype("int")
        prec = (QoS_01 == CPI_01).astype("int").sum() / len(df)
        mse = ((cpi - qos)**2).mean()
        mae = abs((cpi - qos)).mean()
        result["app"].append(app)
        result["01_acc"].append(prec)
        result["mse"].append(mse)
        result["mae"].append(mae)
        result["threshold"].append(j)

result = pd.DataFrame(result)
result.to_csv("worst_cluster_CPI.csv", index=False)
# %%