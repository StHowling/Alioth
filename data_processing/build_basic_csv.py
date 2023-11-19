'''
No merging 3 seconds, retaining timestamp, build the very basic, pure (with only data cleaning and gathering) csv in one .py file
'''

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import re
import shutil

DATADIR = "./data/mul"

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

# Consider all disk and net related metrics as log-keys and ignore them in filtering extremely large outliers (As inspected, there is not such faulty data in them)

LOG_KEYS = ['net_rd_byte',
            'net_wr_byte',
            'net_rd_packet',
            'net_wr_packet',
            'rd_bytes',
            'wr_bytes',
            'rd_total_times',
            'wr_total_times',
            'flush_total_times',
            'tps',
            'rtps',
            'wtps',
            'bread_s',
            'bwrtn_s',
            'rxpck_s',
            'txpck_s',
            'rxkB_s',
            'txkB_s']

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

def get_max_without_outliers(df_counted_add):
    value_keys = list(df_counted_add.columns)
    max_total = {}
    for i in value_keys:
        if i in NONE_SCALE_KEYS or i in LOG_KEYS:
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

def get_stress_index(df, appname):
    stress_type_intensity_to_indices_map = {}
    stress_type = pd.unique(df["stress_type"])
    for i in stress_type:
        stress_type_intensity_to_indices_map[i] = {}
        stress_intensity = pd.unique(df.loc[df["stress_type"] == i, "stress_intensity"])
        # print(stress_type, stress_intensity)
        for j in stress_intensity:
            stress_type_intensity_to_indices_map[i][j] = df.loc[(df["stress_type"] == i) & (df["stress_intensity"] == j)].index

    # Only consider bottom consecutive no_stress index
    indices = stress_type_intensity_to_indices_map["NO_STRESS"][0]
    nostress_starting_idx = len(indices) - 1
    while True:
        if indices[nostress_starting_idx - 1] != indices[nostress_starting_idx] - 1 or nostress_starting_idx == 0:
            break
        else:
            nostress_starting_idx -= 1

    # Use IRQ based method to filter irregular qos value in no_stress
    # milc cannot be filtered because of its large variance
    if appname in ['milc', 'noapp']:
        stress_type_intensity_to_indices_map["NO_STRESS"][0] = indices[nostress_starting_idx:]
    
    else:
        indices = indices[nostress_starting_idx:]
        QoS_label = ["count", "latency", "tps.1", "sent_speed", "received_speed"]
        qos_filter = None
        for qos_label in QoS_label:
            if qos_label in df.columns:
                qos = df.loc[indices,qos_label]
                q1 = qos.quantile(0.2)
                q3 = qos.quantile(0.8)
                irq = (q3 - q1) * 2
                inf = q1 - irq
                sup = q3 + irq
                if qos_filter is None:
                    qos_filter = (qos > inf) & (qos < sup)
                else:
                    qos_filter &= ((qos > inf) & (qos < sup))        

        stress_type_intensity_to_indices_map["NO_STRESS"][0] = qos[qos_filter].index

    return stress_type_intensity_to_indices_map, nostress_starting_idx


def milc_drop_qos(df):
    df_drop = df[df["speed"] < 1.5].copy()
    return df_drop

def data_transform(df, appname, max_total, drop_columns):
    # filter max outliers, set outliers to the resonable max value
    # transform the log_key columns
    '''
        Warning: after careful thoughts, maybe log transform should not be here, rather should be placed after building temporal features
    '''
    df = filter_notcounted(df)
    
    if appname != 'noapp':
        df = add_columns(df)
        df = process_none_scale(df)
        df.drop(drop_columns, axis=1, inplace=True)
    else:
        for i in drop_columns:
            if i in df:
                df.drop(i, axis=1, inplace=True)
    
            
    keys = list(df.columns)
    # df_length = len(df)
    
    for i in keys:
        if i in max_total:
            df.loc[df[i] > max_total[i]["max"], i] = max_total[i]["max"]
            # tmp = (df[i] - max_total[i]["min"]) / (max_total[i]["max"] - max_total[i]["min"])
            # tmp = df[i] / max_total[i]["max"] # min is assumed to be 0
            # tmp[tmp >= 1] = 1.0
        # if i in LOG_KEYS:
        #     tmp = df[i].copy()
        #     tmp = np.log(tmp+1)
        #     df[i] = tmp
    
    df.reset_index(drop=True, inplace=True)

    # Convert raw QoS metric to degradation
    keys = list(df.columns)
    stress_df_index, bottom_no_stress_idx = get_stress_index(df, appname)
    if appname != 'noapp':
        start = keys.index("IPC") + 1
        end = keys.index("stress_type")
        QoS = keys[start: end]
        QoS_nostress = df.loc[stress_df_index["NO_STRESS"][0],QoS].mean()

        df.loc[:, QoS] = df.loc[:, QoS] / QoS_nostress

    # Wrong place in dacbip
    if appname == "milc":
        df = milc_drop_qos(df)

    # drop warm-up phase and intermediate no stress
    nostress = df[df['stress_type'] == "NO_STRESS"]
    nostress_idx = nostress.index
    df.drop(nostress_idx[: bottom_no_stress_idx], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def rename_qos(df, appname, workload):
    rename_map = {
        'cassandra': {'count': 'qos1', 'latency': 'qos2'},
        'etcd': {'count': 'qos1', 'latency': 'qos2'},
        'hbase': {'count': 'qos1', 'latency': 'qos2'},
        'kafka': {'tps.1': 'qos1', 'latency': 'qos2'},
        'milc': {'speed': 'qos1'},
        'mongoDB': {'count': 'qos1', 'latency': 'qos2'},
        'rabbitmq': {'sent_speed':'qos1', 'received_speed':'qos2'},
        'redis': {'count': 'qos1', 'latency': 'qos2'}
    }

    if appname == 'noapp':
        tmp = np.zeros(df.shape[0])
        df.insert(df.shape[1],'qos1',tmp)
        df.insert(df.shape[1],'qos2',tmp)
    else:
        df.rename(columns=rename_map[appname], inplace=True)
    if appname == 'milc':
        tmp = np.zeros(df.shape[0])
        df.insert(df.shape[1],'qos2',tmp)

    app_df = [appname] * df.shape[0]
    workload_df = [workload] * df.shape[0]
    df.insert(df.shape[1], "app", app_df)
    df.insert(df.shape[1], "workload", workload_df)

    return df


if __name__ == '__main__':
    os.chdir(DATADIR)

    # Set up logger

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

    # scan all data file, merge

    csv_file_list = []
    find_all_csv("./", csv_file_list)

    app_csvs = {}
    for i in csv_file_list:
        print("Dealing with {}".format(i))
        dirname = os.path.dirname(i)
        appname = os.path.basename(i).split('-')[0]
        if appname not in app_csvs:
            app_csvs[appname]=[]
        app_csvs[appname].append(i)

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

    # filter "<not" values

    df_counted = filter_notcounted(df)
    df_counted_add = add_columns(df_counted)

    # get extremely large outliers

    max_total = get_max_without_outliers(df_counted_add)
    with open("total_max.json", "w") as f:
        json.dump(max_total, f, indent=4)

    # drop no-volatility columns (e.g. all-zero)
 
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

    # process and merge all distributed files without generating intermediate csv
    df_out = pd.DataFrame()
    outdir = "../output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for f in csv_file_list:
        print("Dealing with {}".format(f))
        df = pd.read_csv(f)
        dirname = os.path.dirname(f)
        workload = os.path.basename(dirname)
        appname = os.path.basename(f).split('-')[0]
        if appname != 'noapp':
            workload_intensity = int(re.search(r'[0-9]+',workload).group())
        else:
            workload_intensity = 0

        df = data_transform(df, appname, max_total, drop_columns)
        df = rename_qos(df, appname, workload_intensity)
        df_out = pd.concat([df_out, df], axis=0)

    df_out.to_csv(os.path.join(outdir,"basic-mul-all-merged.csv"), index=False)
    