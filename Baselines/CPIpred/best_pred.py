# %%
import os
import pandas as pd
import numpy as np
# %%
DATADIR = r"data/mul"
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
loss["01_acc"] = []
loss["app"] = []
loss["threshold"] = []
for k in np.arange(1.05, 1.3, 0.05):
    for app in app_csvs:
        print(app)
        prec = 0
        mse = 0
        mae = 0
        app_len = 0
        for i in app_csvs[app]:
            df = pd.read_csv(i)
            CPI_nostress = df[df["stress_type"] == "NO_STRESS"]["VM_CPI"].mean()
            df = df[~(df["stress_type"] == "NO_STRESS")]
            df_CPI = df["VM_CPI"] / CPI_nostress

            df_keys = list(df.columns)
            QoS = df_keys[df_keys.index("IPC") + 1: df_keys.index("stress_type")]
            if app == "milc":
                df_QoS = 2 - df[QoS[0]]
            elif app == "rabbitmq":
                tmp = (df[QoS[0]] + df[QoS[1]]) / 2
                tmp.name = "speed"
                df_QoS = 2 - tmp
            else:
                df_QoS = df["latency"]
            print(df_QoS)

            lc_01 = (df_QoS >= k).astype("int")
            CPI_01 = (df_CPI >= k).astype("int")
            prec += (lc_01 == CPI_01).astype("int").sum()
            app_len += len(df)
            mse += ((df_QoS - df_CPI)**2).sum()
            mae += abs(df_QoS - df_CPI).sum()
        loss["app"].append(app)
        loss["01_acc"].append(prec / app_len)
        loss["mse"].append(mse / app_len)
        loss["mae"].append(mae / app_len)
        loss["threshold"].append(k)
        print(app_len)
# %%
df = pd.DataFrame(loss)
df.to_csv("best_pred_loss.csv", index=False)

# %%
