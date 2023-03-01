# %%
import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# %%
# DATADIR = r"E:\research\cloudVM\code\data\interference_data\mul"
# OUTPUT_DIR = r"E:\research\cloudVM\code\data\interference_data\monitorless_output\raw"
DATADIR = r"../data/mul"
OUTPUT_DIR = r"../data/monitorless_output/raw"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

NONE_SCALE_KEYS = ['timestamp',
                 'cpu_time',
                 'system_time',
                 'user_time',
                 'mem_util',
                 'user',
                 'nice',
                 'system',
                 'iowait',
                 'steal',
                 'idle',
                 'memused_percentage',
                 'commit_percentage']

LOG_KEYS = ['net_rd_byte',
            'net_wr_byte',
            'rd_bytes',
            'wr_bytes']

logfile = "./get_max.log"
logger = logging.getLogger("get max logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


# %%
def find_all_csv(relative_path, file_list, file_name="-1.csv"):
    for i in os.listdir(relative_path):
        file_path = os.path.join(relative_path, i)
        if os.path.isdir(file_path):
            find_all_csv(file_path, file_list, file_name)
        else:
            if i.endswith(file_name):
                file_list.append(file_path)

def filter_notcounted(df):
    value_keys = list(df.columns)
    if "stress_type" in value_keys:
        idx = value_keys.index("stress_type")
        value_keys = value_keys[: idx]
    if "timestamp" in value_keys:
        idx = value_keys.index("timestamp") + 1
        value_keys = value_keys[idx:]
    filter = (df != "<not")
    df_counted = df[filter]
    df_counted.dropna(axis=0, inplace=True)
    value_astype = {i: "float64" for i in value_keys}
    df_counted = df_counted.astype(value_astype)
    logger.info("After deleting not counted shape: {}".format(df_counted.shape))
    return df_counted

def add_columns(df_counted):
    # Add mem_util
    df_counted_add = df_counted.copy()
    logger.info("Add VM mem util, CPI, RCPI, MPKI")
    mem_util = (df_counted_add["available"] - df_counted_add["unused"]) / df_counted_add["available"]
    df_counted_add.drop(labels=["actual", "available", "unused"], axis=1, inplace=True)
    idx = df_counted_add.columns.get_loc("user_time") + 1
    df_counted_add.insert(idx, "mem_util", mem_util)
    # Add VM CPI, RCPI, MPKI
    VM_CPI = df_counted_add["UNHALTED_CORE_CYCLES"] / df_counted_add["INSTRUCTION_RETIRED"]
    VM_CPI[VM_CPI >= 30] = 30
    VM_RCPI = df_counted_add["UNHALTED_REFERENCE_CYCLES"] / df_counted_add["INSTRUCTION_RETIRED"]
    VM_RCPI[VM_RCPI >= 30] = 30
    VM_MPKI_LLC = df_counted_add["LLC_MISSES"] * 1000.0 / df_counted_add["INSTRUCTION_RETIRED"]
    VM_MPKI_LLC[VM_MPKI_LLC >= 30] = 30
    VM_MPKI_L2 = df_counted_add["L2_RQSTS:MISS"] * 1000.0 / df_counted_add["INSTRUCTION_RETIRED"]
    VM_MPKI_L2[VM_MPKI_L2 >= 300] = 300
    VM_MPKI_L1D = df_counted_add["MEM_LOAD_RETIRED:L1_MISS"] * 1000.0 / df_counted_add["INSTRUCTION_RETIRED"]
    VM_MPKI_L1D[VM_MPKI_L1D >= 300] = 300

    idx = df_counted_add.columns.get_loc("MEM_LOAD_RETIRED:L1_HIT")
    df_counted_add.insert(idx, "VM_CPI", VM_CPI)
    df_counted_add.insert(idx, "VM_RCPI", VM_RCPI)
    df_counted_add.insert(idx, "VM_MPKI_LLC", VM_MPKI_LLC)
    df_counted_add.insert(idx, "VM_MPKI_L2", VM_MPKI_L2)
    df_counted_add.insert(idx, "VM_MPKI_L1D", VM_MPKI_L1D)
    df_counted_add.dropna(axis=0, inplace=True)
    return df_counted_add

def add_binary(df):
    # Add CPU Mem binary features
    logger.info("Add VM cpu binary")
    k = list(df.keys()).index("cpu_time") + 1
    a = df["cpu_time"]
    tmp = (a > 0.95).astype(int)
    df.insert(k, "cpu_g95_binary", tmp)
    tmp = (a > 0.9).astype(int)
    df.insert(k, "cpu_g90_binary", tmp)
    tmp = (a > 0.8).astype(int)
    df.insert(k, "cpu_g80_binary", tmp)
    tmp = ((a <= 0.8) & (a > 0.5)).astype(int)
    df.insert(k, "cpu_g50le80_binary", tmp)
    tmp = (a < 0.5).astype(int)
    df.insert(k, "cpu_l50_binary", tmp)

    logger.info("Add VM mem binary")
    k = list(df.keys()).index("mem_util") + 1
    a = df["mem_util"]
    tmp = (a > 0.8).astype(int)
    df.insert(k, "mem_g80_binary", tmp)
    tmp = ((a <= 0.8) & (a > 0.5)).astype(int)
    df.insert(k, "mem_g50le80_binary", tmp)
    tmp = (a < 0.5).astype(int)
    df.insert(k, "mem_l50_binary", tmp)
    
    logger.info("Add PM cpu binary")
    k = list(df.keys()).index("idle") + 1
    a = 1 - df["idle"]
    tmp = (a > 0.95).astype(int)
    df.insert(k, "PMcpu_g95_binary", tmp)
    tmp = (a > 0.9).astype(int)
    df.insert(k, "PMcpu_g90_binary", tmp)
    tmp = (a > 0.8).astype(int)
    df.insert(k, "PMcpu_g80_binary", tmp)
    tmp = ((a <= 0.8) & (a > 0.5)).astype(int)
    df.insert(k, "PMcpu_g50le80_binary", tmp)
    tmp = (a < 0.5).astype(int)
    df.insert(k, "PMcpu_l50_binary", tmp)

    logger.info("Add PM mem binary")
    k = list(df.keys()).index("memused_percentage") + 1
    a = df["memused_percentage"]
    tmp = (a > 0.8).astype(int)
    df.insert(k, "PMmem_g80_binary", tmp)
    tmp = ((a <= 0.8) & (a > 0.5)).astype(int)
    df.insert(k, "PMmem_g50le80_binary", tmp)
    tmp = (a < 0.5).astype(int)
    df.insert(k, "PMmem_l50_binary", tmp)

    return df


def log_scale(df):
    for i in LOG_KEYS:
        df[i] = df[i].apply(lambda x: np.log(x) if x != 0 else x)
    return

def get_max_without_outliers(df_counted_add):
    value_keys = list(df_counted_add.columns)
    max_total = {}
    for i in value_keys:
        if i in NONE_SCALE_KEYS:
            continue
        max_total[i] = {}
        max_total[i]["outliers"] = []
        col = df_counted_add[i].copy()
        while True:
            max = col.max()
            Q3 = col.quantile(0.75)
            if max <= Q3 * 1e5 or max <= 1e10:
                break
            else:
                max_total[i]["outliers"].append(max)
                col[col.idxmax()] = 0
        max_total[i]["max"] = col.max()
        max_total[i]["min"] = col.min()
        if max_total[i]["max"] > 1e13:
            logger.info("Warning!! {} have a very large value {:e}".format(i, max_total[i]["max"]))
    return max_total

def process_none_scale(df):
    df_proc = df.copy()
    for keys in NONE_SCALE_KEYS:
        if '_time' in keys:
            df_proc[keys] = df_proc[keys]/400.
    return df_proc

def get_stress_index(df):
    stress_index = {}
    stress_type = list(df["stress_type"].drop_duplicates())
    for i in stress_type:
        stress_index[i] = {}
        stress_intensity = list(df[df["stress_type"] == i]["stress_intensity"].drop_duplicates())
        # print(stress_type, stress_intensity)
        for j in stress_intensity:
            stress_index[i][j] = df[(df["stress_type"] == i) & (df["stress_intensity"] == j)].index
    # Only consider bottom no_stress index
    idx = stress_index["NO_STRESS"][0]
    i = len(idx) - 1
    while True:
        if idx[i - 1] != idx[i] - 1:
            break
        else:
            i -= 1
    stress_index["NO_STRESS"][0] = idx[i+4:i+19]
    return stress_index


# %%
csv_file_list = []
find_all_csv(DATADIR, csv_file_list)
# %%
# Read all files
df = pd.DataFrame()
for i in csv_file_list:
    if "noapp" in i:
        continue
    print("Load {}".format(i))
    tmp = pd.read_csv(i)
    keys = list(tmp.columns)
    idx = keys.index("IPC") + 1
    tmp = tmp[keys[1 : idx]]
    df = pd.concat([df, tmp])
print("Total df shape: {}".format(df.shape))

# %%
df_counted = filter_notcounted(df)
df_counted_add = add_columns(df_counted)
max_total = get_max_without_outliers(df_counted_add)
# %%
drop_columns = []
for i in df_counted_add:
    if i in max_total:
        if (max_total[i]["max"] - max_total[i]["min"]) < 1e-8:
            drop_columns.append(i)
    if i in NONE_SCALE_KEYS:
        k_max = df_counted_add[i].max()
        k_min = df_counted_add[i].min()
        if (k_max - k_min) < 1e-8:
            drop_columns.append(i)
print("drop: ", drop_columns)
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
        df = filter_notcounted(df)
        df = add_columns(df)
        df = process_none_scale(df)
        df.drop(drop_columns, axis=1, inplace=True)
        df.drop(['timestamp'], axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        stress_index = get_stress_index(df)

        df_keys = list(df.columns)
        QoS = df_keys[df_keys.index("IPC") + 1: df_keys.index("stress_type")]
        QoS_nostress = df.loc[stress_index["NO_STRESS"][0]][QoS].mean()
        df = df[~(df['stress_type']=='NO_STRESS')]
        for j in df_keys:
            if j not in max_total:
                continue
            df[j][df[j] > max_total[j]["max"]] = max_total[j]["max"]
        df = add_binary(df)
        log_scale(df)

        df.reset_index(drop=True, inplace=True)
        df_keys = list(df.columns)
        QoS = df_keys[df_keys.index("IPC") + 1: df_keys.index("stress_type")]
        df[QoS] = df[QoS] / QoS_nostress
        df_all = pd.concat([df_all, df], axis=0)

    df_all.reset_index(drop=True, inplace=True)
    df_keys = list(df_all.columns)
    value_keys = df_keys[: df_keys.index("IPC") + 1]
    QoS = df_keys[df_keys.index("IPC") + 1: df_keys.index("stress_type")]
    # Deal with app QoS
    if app == "milc":
        df_QoS = 1 - df_all[QoS[0]]
        # df_QoS[df_QoS > 1.2] = 1.2
    elif app == "rabbitmq":
        tmp = (df_all[QoS[0]] + df_all[QoS[1]]) / 2
        tmp.name = "speed"
        df_QoS = 1 - tmp
    else:
        df_QoS = df_all["latency"] - 1
    df_QoS.name = "QoS"
    logger.info("df_QoS name: {}".format(df_QoS.name))
    # Standard normalization
    df_values = df_all[value_keys]
    scaler = StandardScaler()
    np_values = scaler.fit_transform(df_values)
    df_tmp = pd.DataFrame(np_values, columns=value_keys)
    df_tmp = pd.concat([df_tmp, df_QoS], axis=1)
    df_tmp.to_csv(os.path.join(OUTPUT_DIR, app + ".csv"), index=False)
    df_total = pd.concat([df_total, df_tmp], axis=0)
df_total.to_csv(os.path.join(OUTPUT_DIR, "total.csv"), index=False)
logger.info("Done!")

# %%
# np.log(df["wr_bytes"])
# %%
# csv_file_list = []
# find_all_csv(OUTPUT_DIR, csv_file_list, ".csv")
# df_total = pd.DataFrame()
# for i in csv_file_list:
#     if "milc" in i:
#         df = pd.read_csv(i)
#         QoS = df_keys[df_keys.index("IPC") + 1: df_keys.index("stress_type")]
#         df_QoS = 1 - df[QoS]
#         df_total = pd.concat([df_total, df_QoS], axis = 0)
# # %%
# a = pd.DataFrame({"a": [1, 2, 3]})
# b = pd.DataFrame({"b": [1, 2, 3]})
# pd.concat([a, b], axis=0)
# %%
