import numpy as np
import pandas as pd



    # '0':['hbase','cassandra','mongoDB'],
    # '1':['kafka','rabbitmq'],
    # '2':['etcd','redis'],
    # '3':['milc'],

dirpre1 = '/home/yyx/interference_prediction/Alioth/Result/DAE_Predcassandra.npy'
VM_data_no_stress1 = np.load(dirpre1, allow_pickle=True)
print(np.shape(VM_data_no_stress1))

dirpre2 = '/home/yyx/interference_prediction/Alioth/Result/DAE_Predhbase.npy'
VM_data_no_stress2 = np.load(dirpre2, allow_pickle=True)
print(np.shape(VM_data_no_stress2))

dirpre3 = '/home/yyx/interference_prediction/Alioth/Result/DAE_PredmongoDB.npy'
VM_data_no_stress3 = np.load(dirpre3, allow_pickle=True)
print(np.shape(VM_data_no_stress3))

dirpre4 = '/home/yyx/interference_prediction/Alioth/Result/DAE_Predkafka.npy'
VM_data_no_stress4 = np.load(dirpre4, allow_pickle=True)
print(np.shape(VM_data_no_stress4))

dirpre5 = '/home/yyx/interference_prediction/Alioth/Result/DAE_Predrabbitmq.npy'
VM_data_no_stress5 = np.load(dirpre5, allow_pickle=True)
print(np.shape(VM_data_no_stress5))

dirpre6 = '/home/yyx/interference_prediction/Alioth/Result/DAE_Predetcd.npy'
VM_data_no_stress6 = np.load(dirpre6, allow_pickle=True)
print(np.shape(VM_data_no_stress6))

dirpre7 = '/home/yyx/interference_prediction/Alioth/Result/DAE_Predredis.npy'
VM_data_no_stress7 = np.load(dirpre7, allow_pickle=True)
print(np.shape(VM_data_no_stress7))

dirpre8 = '/home/yyx/interference_prediction/Alioth/Result/DAE_Predmilc.npy'
VM_data_no_stress8 = np.load(dirpre8, allow_pickle=True)
print(np.shape(VM_data_no_stress8))


data = np.vstack((VM_data_no_stress6,VM_data_no_stress1,VM_data_no_stress4,VM_data_no_stress8,VM_data_no_stress2,VM_data_no_stress3,VM_data_no_stress5))
print(np.shape(data))
np.save('/home/yyx/interference_prediction/Alioth/DATA/DAE_Pred-redis.npy',data)