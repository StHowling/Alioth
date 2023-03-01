# %%
import os
import pandas as pd
# %%
DATADIR = r"E:\research\cloudVM\code\data\interference_data\mul"
OUTPUT_DIR = r"E:\research\cloudVM\code\data\interference_data\practical"
# DATADIR = r"../data/mul"
# OUTPUT_DIR = r"../data/pratical"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
df_total = pd.DataFrame()
for app in app_csvs:
    df_all = pd.DataFrame()
    for i in app_csvs[app]:
        print("Dealing with {}".format(i))
        df = pd.read_csv(i)
        df.drop(["timestamp"], axis=1, inplace=True)
        df_keys = list(df.columns)
        QoS = df_keys[df_keys.index("IPC") + 1: df_keys.index("stress_type")]
        df_values = df.iloc[:, : df_keys.index("IPC") + 1]
        if app == "milc":
            df_QoS = 2 - df[QoS[0]]
        elif app == "rabbitmq":
            tmp = (df[QoS[0]] + df[QoS[1]]) / 2
            tmp.name = "speed"
            df_QoS = 2 - tmp
        else:
            df_QoS = df["latency"]
        df_QoS.name = "QoS"
        df_tmp = pd.concat([df_values, df_QoS], axis=1)
        df_all = pd.concat([df_all, df_tmp], axis=0)
    df_all.to_csv(os.path.join(OUTPUT_DIR, app + ".csv"), index=False)
    df_total = pd.concat([df_total, df_all], axis=0)
df_total.to_csv(os.path.join(OUTPUT_DIR, "total.csv"), index=False)

# %%
