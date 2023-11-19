import pandas as pd
import numpy as np
import os
from typing import List

DATADIR = "./data/output"

RAW_FEATURE_NUM = 218 
LIBVIRT_FEATURE_NUM = 15
SAR_FEATURE_NUM = 24
PQOS_FEATURE_NUM = 4

ALIOTH_OLD_KEYS = [
    'system_time', 'idle', 'MBL', 'user', 'mem_util', 'cpu_time', 'net_wr_packet',
    'L2_RQSTS:DEMAND_DATA_RD_HIT', 'LLC', 'VM_RCPI', 'MEM_LOAD_L3_MISS_RETIRED:REMOTE_DRAM',
    'UNC_C_REQUESTS:READS', 'net_rd_packet', 'DTLB-PREFETCHES', 'MEM_LOAD_RETIRED:L3_MISS',
    'kbbuffers', 'MEM_LOAD_L3_MISS_RETIRED:LOCAL_DRAM', 'RS_EVENTS:EMPTY_CYCLES',
    'net_wr_byte', 'MBR', 'net_rd_byte', 'IPC'
]

def build_DAE_features(df, save_name, feature_columns:List=None):
    label_columns = ['app','workload','stress_type','stress_intensity','interval']
    if feature_columns is None:
        feature_columns = df.columns[1:-7]
    else:
        feature_columns = set(feature_columns).intersection(set(df.columns))
    
    apps = pd.unique(df['app'])
    df_out=pd.DataFrame()
    for app in apps:
        workload_levels = pd.unique(df.loc[df['app']==app, 'workload'])
        for wl in workload_levels:
            stress_metrics = df.loc[((df['app']==app) & (df['workload']==wl) & (df['stress_type']!='NO_STRESS')),feature_columns].reset_index(drop=True)
            aux_labels = df.loc[((df['app']==app) & (df['workload']==wl) & (df['stress_type']!='NO_STRESS')),label_columns].reset_index(drop=True)
            no_stress_metrics = df.loc[((df['app']==app) & (df['workload']==wl) & (df['stress_type']=='NO_STRESS')),feature_columns].rename(columns={f:f+'.nostress' for f in feature_columns})
            if no_stress_metrics.shape[0] == 1:
                no_stress_metrics = pd.concat([no_stress_metrics for i in range(stress_metrics.shape[0])],axis=0).reset_index(drop=True)
                tmp = pd.concat([stress_metrics,no_stress_metrics,aux_labels],axis=1)
                df_out = pd.concat([df_out,tmp],axis=0)
            else:
                for j in range(no_stress_metrics.shape[0]):
                    cur_row = no_stress_metrics.iloc[j,:]
                    cur_row_expansion = pd.concat([cur_row for i in range(stress_metrics.shape[0])],axis=0)
                    tmp = pd.concat([stress_metrics,cur_row_expansion,aux_labels],axis=1)
                    df_out = pd.concat([df_out,tmp],axis=0)

    df_out.to_csv(save_name,index=False)

def get_online_feature_columns(cols):
    num_feature_round = len(cols) // RAW_FEATURE_NUM
    feature_columns = []
    for i in range(num_feature_round):
        feature_columns = feature_columns + cols[1 + i*RAW_FEATURE_NUM : 1 + i*RAW_FEATURE_NUM + LIBVIRT_FEATURE_NUM ].tolist()
        feature_columns = feature_columns + cols[1 + (i+1)*RAW_FEATURE_NUM - PQOS_FEATURE_NUM - SAR_FEATURE_NUM : 1 + (i+1)*RAW_FEATURE_NUM - PQOS_FEATURE_NUM ].tolist()
    return feature_columns


if __name__ == '__main__':
    os.chdir(DATADIR)

    # original flavor of data preprocessing
    df = pd.read_csv('mul_all_int3_nowarming_nosliding_mean.csv')

    # No feature selection
    build_DAE_features(df, 'DAE_features_all_int3_nowarming_nosliding_mean.csv')

    # Online feature selection
    feature_columns = get_online_feature_columns(df.columns)
    
    build_DAE_features(df, 'DAE_features_online_int3_nowarming_nosliding_mean.csv', feature_columns=feature_columns)

    # all-included no sliding
    df = pd.read_csv('mul_all_intall_warming3_nosliding_all.csv').drop_duplicates()
    feature_columns = get_online_feature_columns(df.columns)
    build_DAE_features(df, 'DAE_features_all_intall_warming3_nosliding_all.csv')
    build_DAE_features(df, 'DAE_features_online_intall_warming3_nosliding_all.csv', feature_columns=feature_columns)

