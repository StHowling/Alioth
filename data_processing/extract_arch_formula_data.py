# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import sys
import logging
import pandas as pd
import re
import json

# %%
# DATADIR = r"E:\research\cloudVM\code\data\interference_data\mul"
DATADIR = r"./mul"
os.chdir(DATADIR)


# %%
# setting up logger
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


# %%
def find_all_csv(relative_path, file_list, file_name="-1.csv"):
    for i in os.listdir(relative_path):
        file_path = os.path.join(relative_path, i)
        if os.path.isdir(file_path):
            find_all_csv(file_path, file_list, file_name)
        else:
            if i.endswith(file_name):
                file_list.append(file_path)

def find_intensity(i):
    directories=i.split('/')
    for item in directories:
        if 'wl' in item:
            return int(re.search(r'[0-9]+',item).group())

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
find_all_csv("./", csv_file_list)


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
    # if (os.path.basename(i).split('-')[0]=='noapp'):
    #     print(df)
    #     break
print("Total df shape: {}".format(df.shape))

# %%
# Get the max of all files
df_counted = filter_notcounted(df)
df_counted_add = add_columns(df_counted)
max_total = get_max_without_outliers(df_counted_add)
with open("total_max.json", "w") as f:
    json.dump(max_total, f, indent=4)
# %%
drop_columns = []
for i in df_counted:
    if i in max_total:
        if (max_total[i]["max"] - max_total[i]["min"]) < 1e-8:
            drop_columns.append(i)
    if i in NONE_SCALE_KEYS:
        k_max = df_counted[i].max()
        k_min = df_counted[i].min()
        if (k_max - k_min) < 1e-8:
            drop_columns.append(i)
# %%

for f in csv_file_list:
    print("Dealing with {}".format(f))
    df = pd.read_csv(f)
    dirname = os.path.dirname(f[2:])
    dirname = dirname.replace('\\','/')
    outname = os.path.basename(f).split('-')[0] + "-extract-af-features.csv"
    noapp = (os.path.basename(f).split('-')[0]=='noapp')
    #print(dirname, outname)
    df = filter_notcounted(df)
   
    
    if not noapp:
        df = add_columns(df)
        df = process_none_scale(df)
        df.drop(drop_columns, axis=1, inplace=True)
    else:
        continue
        for i in drop_columns:
            if i in df:
                df.drop(i, axis=1, inplace=True)

    vm_workload_features=['timestamp','cpu_time','mem_util','net_rd_byte','net_wr_byte','net_rd_packet','net_wr_packet','VM_CPI']
    arch_formula_features=['UNHALTED_REFERENCE_CYCLES','UNHALTED_CORE_CYCLES','CYCLE_ACTIVITY:STALLS_TOTAL']
    keys = list(df.columns)
    labels = keys[keys.index("IPC") + 1:]

    for i in keys:
        if i not in arch_formula_features and i not in labels and i not in vm_workload_features:
            df.drop(i, axis=1, inplace=True)            
    
    # eliminating extreme values
    for i in vm_workload_features:
        if i not in max_total:
            continue
        tmp = (df[i] - max_total[i]["min"]) / (max_total[i]["max"] - max_total[i]["min"])
        # tmp = df[i] / max_total[i]["max"] # min is assumed to be 0
        tmp[tmp >= 1] = 1.0
        df[i] = tmp

    for i in arch_formula_features:
        if i not in max_total:
            continue
        tmp = df[i]
        tmp[tmp >= max_total[i]["max"]] = max_total[i]["max"]
        df[i] = tmp
    
    df.reset_index(drop=True, inplace=True)
    if 'milc' in f:
        df['speed']=1/df['VM_CPI']
        
    keys = list(df.columns)
    df_out = pd.DataFrame()
    STEP = 3
    stress_index = get_stress_index(df)
    if noapp:
        STEP=30
    else:
        start = keys.index("UNHALTED_CORE_CYCLES") + 1
        end = keys.index("stress_type")
        QoS = keys[start: end]
        QoS_nostress = df.loc[stress_index["NO_STRESS"][0]][QoS].mean()
    
    # Merge data point every 3 seconds
    for si in stress_index:
        for j in stress_index[si]:
            k = 0
            while k < len(stress_index[si][j]):
                tmp = df.iloc[stress_index[si][j][k : k + STEP], 1: -2].mean()
                df_tmp = tmp.to_frame().T
                df_tmp.insert(0, "timestamp", df.loc[stress_index[si][j][k]]["timestamp"])
                df_tmp.insert(df_tmp.shape[1], "stress_type", df.loc[stress_index[si][j][k]]["stress_type"])
                df_tmp.insert(df_tmp.shape[1], "stress_intensity", df.loc[stress_index[si][j][k]]["stress_intensity"])
                if not noapp:
                    # Convert QoS to degration
                    df_tmp[QoS] = df_tmp[QoS] / QoS_nostress
                k += STEP
                df_out = pd.concat([df_out, df_tmp])

    if 'milc' in f:
        tmp=df_out['speed']
        tmp[tmp>1.2]=1.2
        df_out['speed']=tmp
    df_out.drop('VM_CPI',axis=1,inplace=True)
    df_out.to_csv(dirname+'/'+outname, index=False)
# %%
csv_file_list = []
find_all_csv("./", csv_file_list, 'extract-af-features.csv')
app_csvs = {}
for i in csv_file_list:
    print("Dealing with {}".format(i))
    dirname = os.path.join(DATADIR, os.path.dirname(i[2:]))
    dirname = dirname.replace('\\','/')
    appname = os.path.basename(i).split('-')[0]
    # if appname == 'noapp':
    #     continue
    if appname not in app_csvs:
        app_csvs[appname]=[]
    app_csvs[appname].append(i)

for app in app_csvs:
    if app == 'noapp':
        continue
    df_all = pd.DataFrame()
    dirname = "../output/Alioth-raw/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    outname = app + "-af-wl_3merged.csv"
    logger.info("Generate {} arch-formula data for Alioth".format(app))
    for i in app_csvs[app]:
        print("Dealing with {}".format(i))
        wl_intensity=find_intensity(i)

        if (app=='kafka' and wl_intensity==20000) or (app=='rabbitmq' and wl_intensity==4):
            continue

        df = pd.read_csv(i)
        df = df.drop(['timestamp'], axis=1)
        df.insert(df.shape[1],"WORKLOAD_INTENSITY",wl_intensity)

        df_all = pd.concat([df_all, df])
    
    df_all.to_csv(os.path.join(dirname, outname), index=False)
# %%
os.chdir('../output/Alioth-raw')
# %%
category_map={
    'cassandra':0,
    'hbase':0,
    'mongoDB':0,        # NoSQL
    'rabbitmq':1,
    'kafka':1,          # Message Middleware
    'etcd':2,
    'redis':2,          # K-V Store / Caching
    'milc':3            # HPC
}

QoS_map={
    'cassandra': ['count', 'latency'],
    'etcd': ['latency', 'count'],
    'hbase': ['count', 'latency'],
    'kafka': ['tps.1', 'latency'],
    'milc': ['speed'],
    'mongoDB': ['count', 'latency'],
    'rabbitmq': ['sent_speed', 'received_speed'],
    'redis': ['latency', 'count']
}

def gen_all_merged_by_category_data(outname,metric='LC',dic=None):
    df_all=pd.DataFrame()
    for file in os.listdir('./'):
        if 'af-wl' not in file or 'all' in file:
            continue
        app=file.split('-')[0]
        print(file)
        df=pd.read_csv(file)
        if app=='milc':
            df.rename(columns={'speed':'QoS'},inplace=True)
        else:
            if app=='rabbitmq':
                QOS=(df['sent_speed']+df['received_speed'])/2
            elif metric=='LC':
                QOS=df['latency']
            else:
                if app!='kafka':
                    QOS=df['count']
                else:
                    QOS=df['tps.1']
            df.drop(QoS_map[app],axis=1,inplace=True)
            df.insert(len(df.columns)-3,"QoS",QOS)

        if dic==None:
            df.insert(df.shape[1],'CATEGORY',app)
        else:
            df.insert(df.shape[1],'CATEGORY',dic[app])
        df_all=pd.concat([df_all,df])
    df_all.to_csv(outname,index=False)


gen_all_merged_by_category_data('af-all_merged_by_app.csv')
gen_all_merged_by_category_data('af-all_merged_by_category.csv',dic=category_map) 
# %%


# %%
