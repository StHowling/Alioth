import subprocess
import sys
import time
import datetime
from multiprocessing import Pool

app_start_map={
    'mysql':   	['systemctl start mysql'],
    'etcd':    	['docker start etcd'],
    'rabbitmq':	['docker start rabbitmq'],
    'mongodb': 	['docker start mongodb'],
    'cassandra':['docker start cassandra'], 
    'hbase':	['docker start hbase'],
    'redis':	['docker start redis'],  
    'kafka':	['nohup /root/software/kafka/bin/zookeeper-server-start.sh -daemon /root/software/kafka/config/zookeeper.properties &'] 
}

app_stop_map={
    'mysql':   	['systemctl stop mysql'],
    'etcd':    	['docker kill etcd'],
    'rabbitmq':	['docker kill rabbitmq'],
    'mongodb': 	['docker kill mongodb'],
    'cassandra':['docker kill cassandra'],
    'hbase':	['docker kill hbase'], 
    'redis':	['docker kill redis'],  
    'kafka':	['/root/kill-kafka.sh'] 
}

def test_kafka_up(flag):
    cmd_evidence=[b'java -Xmx512M',b'java -Xmx1G']
    p = subprocess.Popen("ps -aux",shell=True,stdout=subprocess.PIPE)
    plist = p.stdout.readlines()
    for item in plist:
        if cmd_evidence[flag] in item:
            return True
    return False


if __name__ == '__main__':
    if sys.argv[2] == '0':
        if sys.argv[1] != 'kafka' and sys.argv[1] != 'hbase':
            p = subprocess.Popen(app_start_map[sys.argv[1]], shell=True)
            p.wait()
        elif sys.argv[1] == 'hbase':
            while True:
                APP_UP = 0
                p = subprocess.Popen(app_start_map[sys.argv[1]], shell=True)
                p.wait()

                time.sleep(10)

                p = subprocess.Popen("docker ps",shell=True,stdout=subprocess.PIPE)
                plist = p.stdout.readlines()
                for item in plist:
                    if b'hbase' in item:
                        APP_UP = 1
                        break
                if APP_UP == 1:
                    break 
        else:
            while True:
                if not test_kafka_up(0):
                    p = subprocess.Popen(app_start_map[sys.argv[1]], shell=True)
                    p.wait()
                    print(0)
                    time.sleep(5)

                if not test_kafka_up(1):
                    p = subprocess.Popen(['/root/software/kafka/bin/kafka-server-start.sh -daemon /root/software/kafka/config/server.properties'], shell=True)
                    p.wait()
                    print(1)
                    time.sleep(10)

                time.sleep(10)
                if test_kafka_up(0) and test_kafka_up(1):
                    break
    else: 
        p = subprocess.Popen(app_stop_map[sys.argv[1]], shell=True)
        p.wait()
