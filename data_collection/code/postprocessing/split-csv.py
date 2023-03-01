import numpy as np
import pandas as pd
import os

APP_REDIS = 'redis'
APP_STORE = 'ab'
APP_MYSQL = 'sysbench'
APP_RABBITMQ = 'rabbitmq'
APP_ETCD = 'etcd'
APP_KAFKA = 'kafka'
APP_MONGODB = 'mongoDB'
APP_FFMPEG = 'ffmpeg'
APP_HBASE = 'hbase'
APP_CASSANDRA = 'cassandra'
APP_SPARK = 'spark'

def split_out_micro(src_file, dst_file, pid):
    df=pd.read_csv(src_file,sep=';')
    unrelated=[]
    for i in range(len(df.index)):
        if df.at[i,'pid']!=pid:
            unrelated.append(i)
    df_new=df.drop(unrelated)
    df_new.to_csv(dst_file, sep=';', index=False)

def split_out_macro(src_file, dst_file, vm_name):
    df=pd.read_csv(src_file)
    unrelated=[]
    for i in range(len(df.index)):
        if df.at[i,'name']!=vm_name:
            unrelated.append(i)
    df_new=df.drop(unrelated)
    df_new.to_csv(dst_file, index=False)

src_dir="D:/ML-data/4-colocation-0409/run-2"
app_list=[APP_FFMPEG, APP_ETCD, APP_REDIS, APP_RABBITMQ]
pid_list=[29471, 47954, 32919, 17386]
vm_list=['instance-0000074a', 'instance-0000074b',
        'instance-0000074c', 'instance-0000074d']

for filename in os.listdir(src_dir):
    path=os.path.join(src_dir,filename)
    if "events.csv" in filename:
        for i in range(len(app_list)):
            output_path=os.path.join(src_dir,app_list[i],filename)
            split_out_micro(path,output_path,pid_list[i])
    elif "metrics.csv" in filename:
        for i in range(len(app_list)):
            output_path=os.path.join(src_dir,app_list[i],filename)
            split_out_macro(path,output_path,vm_list[i])
    elif "events.log" in filename:
        for i in range(len(app_list)):
            output_path=os.path.join(src_dir,app_list[i],filename)
            with open(output_path,'w') as f:
                rf=open(path,'r')
                f.writelines(rf.readlines())
                rf.close()

            

