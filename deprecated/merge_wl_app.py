# After running data_generate.py
# %%
import re
import os
import sys
import logging
import numpy as np
import pandas as pd

DATADIR='./mul/' # ensure the relative path
os.chdir(DATADIR)

logfile='./merge_wl_app.log'
logger = logging.getLogger("merge loads logger")
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

# %%
csv_file_list = []
find_all_csv("./", csv_file_list, '3merge.csv')
# %%
app_csvs = {}
for i in csv_file_list:
    print("Dealing with {}".format(i))
    dirname = os.path.dirname(i)
    dirname = dirname.replace('\\','/')
    appname = os.path.basename(i).split('-')[0]
    # if appname == 'noapp':
    #     continue
    if appname not in app_csvs:
        app_csvs[appname]=[]
    app_csvs[appname].append(i)

# %%
for app in app_csvs:
    if app == 'noapp':
        continue
    df_all = pd.DataFrame()
    dirname = "../output/Alioth-raw/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    outname = app + "-wl_3merged.csv"
    logger.info("Generate {} raw data for Alioth".format(app))
    for i in app_csvs[app]:
        print("Dealing with {}".format(i))
        wl_intensity=find_intensity(i)

        if (app=='kafka' and wl_intensity==20000) or (app=='rabbitmq' and wl_intensity==4):
            continue

        df = pd.read_csv(i)
        df = df.drop(['timestamp'], axis=1)
        df.insert(df.shape[1],"WORKLOAD_INTENSITY",wl_intensity)

        df_all = pd.concat([df_all, df])
    df_all_keys = list(df_all.columns)
    vmr_index = df_all_keys.index("net_wr_packet")
    vm_he_index = df_all_keys.index("UNHALTED_CORE_CYCLES")
    pm_index = df_all_keys.index("IPC")
    logger.info("\tvm resource data index: 0 : {}".format(vmr_index + 1))
    logger.info("\tvm hardware events data index: {} : {}".format(vmr_index + 1, vm_he_index + 1))
    logger.info("\tpm data index: {} : {}".format(vm_he_index + 1, pm_index + 1))
    logger.info("\tlabel and meta info index: {} :".format(pm_index + 1))
    df_all.to_csv(dirname+outname, index=False)

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

# Use latency as default QoS metirc
def gen_all_merged_by_category_data(outname,metric='LC',dic=None):
    df_all=pd.DataFrame()
    for file in os.listdir('./'):
        if 'all' in file or 'af-wl' in file:
            continue
        app=file.split('-')[0]
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


# %%
gen_all_merged_by_category_data('all_merged_by_app.csv')
gen_all_merged_by_category_data('all_merged_by_category.csv',dic=category_map) 

# %%
