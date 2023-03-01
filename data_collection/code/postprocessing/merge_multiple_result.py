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

def merge(src_dir,dst_dir,_app):
    save_name=""
    for filename in os.listdir(src_dir):
        if _app in filename:
            save_name+=filename
            break
    with open (os.path.join(dst_dir,save_name),'w') as f:
        for filename in os.listdir(src_dir):
            if _app in filename:
                rf=open(os.path.join(src_dir,filename),'r')
                f.writelines(rf.readlines())
                rf.close()

src_dir="D:/ML-data/4-colocation-0409/run-2/raw"
dst_dir="D:/ML-data/4-colocation-0409/run-2/" + APP_REDIS
merge(src_dir,dst_dir,APP_REDIS)