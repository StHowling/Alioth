import json
import multiprocessing
import os
import Queue
import random
import requests
import subprocess
import sys
import time
import re

START_FAILED = 'start failed'
STOP_FAILED = 'stop failed'
SUCCESS = 'success'
MIN_STRESS_TIME = 5
MAX_STRESS_TIME = 10
STRESS_TIME = 10

stress_cpunum_map = {
    2: ['192.168.0.110'],
    4: ['192.168.0.128'],
    6: ['192.168.0.110','192.168.0.128'],
    8: ['192.168.0.61'],
    10: ['192.168.0.110','192.168.0.61'],
    12: ['192.168.0.128','192.168.0.61'],
    14: ['192.168.0.110','192.168.0.128','192.168.0.61'],
    16: ['192.168.0.170'],
    18: ['192.168.0.110','192.168.0.170'],
    20: ['192.168.0.128','192.168.0.170'],
    22: ['192.168.0.110','192.168.0.128','192.168.0.170'],
    24: ['192.168.0.61','192.168.0.170'],
    26: ['192.168.0.110','192.168.0.61','192.168.0.170'],
    28: ['192.168.0.128','192.168.0.61','192.168.0.170'],
    30: ['192.168.0.110','192.168.0.128','192.168.0.61','192.168.0.170'],
    32: ['192.168.0.245'],
    48: ['192.168.0.170','192.168.0.245']
}

workload_host_addr = '71.12.110.25'

stress_pattern = re.compile(r'([a-zA-Z\-]+)')

stress_type_map = ['NET', 'L', 'MBW', 'FIO']
stress_level_map = {'NET':[_ for _ in range(3,16,3)], 'L':[_ for _ in range(11,0,-1)], 'MBW':[_ for _ in range(0, 17)], 'FIO':[_ for _ in range(0, 8)]}

def _wait_and_record_stress(wf, stress_type="0", stress_level="0"):
    wf.write('start stress %s with level %s at %d\n' % (stress_type, stress_level, time.time()))
    duration = STRESS_TIME#random.randint(MIN_STRESS_TIME, MAX_STRESS_TIME)
    #print('stress for %d seconds' % duration)
    time.sleep(duration)
    wf.write('end at %d\n' % time.time())
    return duration


def _stop_stress(base_urls, cmd_name, duration):
    for base_url in base_urls:
        resp = requests.delete('%s?command=%s&force=true' % (
            base_url, cmd_name))
        if resp.status_code != 200:
            if resp.status_code != 403:
                print('failed to stop stress %s' % cmd_name)
                return duration, STOP_FAILED
            elif 'not running' not in resp.text:
                print('failed to stop stress %s' % cmd_name)
                return duration, STOP_FAILED
    return duration, SUCCESS


def random_stress(base_url, cmd_name, retry_stop, wf, stress_code):
    duration = -1
    if not retry_stop:
        # payload = [cmd_name, '-C', '0']
        # payload = [cmd_name, '-S', '0']
        # payload = [cmd_name, '-c', '0']
        # payload = [cmd_name, '-m', '0']
        payload = [cmd_name, '-%s' % stress_code, '0']
        resp = requests.post(base_url, data=json.dumps(payload))
        if resp.status_code != 200:
            print('failed to start stress')
            return duration, START_FAILED
        duration = _wait_and_record_stress(wf)
    return _stop_stress([base_url], cmd_name, duration)



def random_idle(q, stress_duration):
    duration = random.randint(stress_duration * 5, stress_duration * 6)
    print('idle for %d seconds', duration)
    for i in range(duration):
        time.sleep(1)
        try:
            q.get_nowait()
            return True
        except Queue.Empty:
            continue
    return False


def stress_loop(q, stress_code, more_stress, stress_once_idx=None):
    if stress_code == '0':
        return
    current_run_idx = 0
    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    wf = open('%s-stress.log' % current, 'w')
    url = 'http://192.168.0.245:8090/v1/cmd'
    more_url = 'http://192.168.0.170:8090/v1/cmd'
    name = 'stress'
    retry_stop = False
    already_run = False
    last_duration = -1
    stopped = random_idle(q, 5)
    if stopped:
        wf.flush()
        wf.close()
        return
    while True:
        if stress_once_idx is not None:
            if current_run_idx < stress_once_idx:
                # haven't reached the target stress run, just sleep
                duration = random.randint(MIN_STRESS_TIME, MAX_STRESS_TIME)
                time.sleep(duration)
                # handle stop signal sent by main process
                stopped = random_idle(q, duration)
                if stopped:
                    wf.flush()
                    wf.close()
                    break
                # increase current_run_idx and start a new loop
                current_run_idx += 1
                continue
            if current_run_idx > stress_once_idx and (
                    not retry_stop and already_run):
                # already pass the target stress run, we check and ensure the
                # stress has been started then stopped, otherwise, retry logic
                # below will be executed
                wf.flush()
                wf.close()
                break
            # if current_run_idx == stress_once_idx:
            #     <reach the target run, execute run stress logic below>

        # run stress and retry stopping stress if needed, based on the
        # retry_stop flag
        if more_stress:
            duration, res = random_double_stress(
                url, more_url, name, retry_stop, wf, stress_code)
        else:
            duration, res = random_stress(
                url, name, retry_stop, wf, stress_code)
        if res == STOP_FAILED:
            retry_stop = True
        else:
            retry_stop = False
        if duration != -1:
            last_duration = duration
            already_run = True
        if res == SUCCESS:
            stopped = random_idle(q, last_duration)
            if stopped:
                wf.flush()
                wf.close()
                break
            last_duration = -1
        else:
            time.sleep(1)
        current_run_idx += 1

def stress_constant_start(stress_code, stress_vm=16):
    if stress_code == '0':
        return SUCCESS
    stressors = stress_code.split(',')
    stress_type = []
    stress_workers = {}
    for i in range(len(stressors)):
        s = stress_pattern.match(stressors[i]).group(0)
        if s=='L':
            bits_code = stressors[i][len(s):]
            if (len(bits_code) <= 2): #support #ways -> hex
                num_w = int(bits_code)
                bits_code = "%#x"%(((1 << (num_w))-1) << (11-num_w))
            #print(bits_code)
            base_url = 'http://%s:8090/v1/cmd' % workload_host_addr
            payload = ["CAT", "/home/fsp/czy/modify_cacheways.py", bits_code]
            resp = requests.post(base_url, data=json.dumps(payload))
            #print(resp)
            if resp.status_code != 200:
                print('failed to modify cacheways')
                return START_FAILED
        elif s=='MBW':
            base_url = 'http://%s:8090/v1/cmd' % workload_host_addr # restrict stress VM to use the least LLC
            payload = ["CATstress", "/home/fsp/czy/modify_cacheways.py", "0x1", "1"]
            resp = requests.post(base_url, data=json.dumps(payload))
            #print(resp)
            if resp.status_code != 200:
                print('failed to modify cacheways')
                return START_FAILED
            
            workers = stressors[i][len(s):]
            base_url = 'http://%s:8090/v1/cmd' % stress_cpunum_map[stress_vm][0]
            payload = ["MBW", "/root/mbw/start_mbw.py", workers]
            resp = requests.post(base_url, data=json.dumps(payload))
            #print(resp)
            if resp.status_code != 200:
                print('failed to start mbw')
                return START_FAILED
        elif s=='FIO':
            depths = stressors[i][len(s):]
            if int(depths)==0: continue
            base_url = 'http://%s:8090/v1/cmd' % stress_cpunum_map[stress_vm][0]
            payload = ["FIO", "/root/start_fio.py", depths]
            resp = requests.post(base_url, data=json.dumps(payload))
            time.sleep(1)
            if resp.status_code != 200:
                print('failed to start fio')
                return START_FAILED
        elif s=='CU':
            putil = stressors[i][len(s):] # in percentage
            base_url = 'http://%s:8090/v1/cmd' % workload_host_addr
            payload = ["CU", "/home/fsp/czy/ch_cpu_util/ch_cpu.py", putil]
            resp = requests.post(base_url, data=json.dumps(payload))
            #print(resp)
            if resp.status_code != 200:
                print('failed to modify cpu utilization rate limit')
                return START_FAILED
        elif s=='NET':
            p = subprocess.Popen(['iperf3', '-s'])
            bandwidth = stressors[i][len(s):] #
            base_url = 'http://%s:8090/v1/cmd' % stress_cpunum_map[stress_vm][0]
            payload = ["NET", "/root/start_iperf3.py", bandwidth]
            resp = requests.post(base_url, data=json.dumps(payload))
            #print(resp)
            if resp.status_code != 200:
                print('failed to start iperf')
                return START_FAILED
        elif len(s) != 1:
            stress_type.append('-'+s)
            stress_workers['-'+s] = stressors[i][len(s):].split('.')
        else:
            stress_type.append(s)
            stress_workers[s] = stressors[i][len(s):].split('.')
    
    if (len(stress_type) == 0):
        return SUCCESS
    cmd_name = 'stress'
    stress_vm_list = stress_cpunum_map[stress_vm]
    for i in range(len(stress_vm_list)):
        payload = [cmd_name]
        for j in range(len(stress_type)):
            payload.append('-%s' % stress_type[j])
            payload.append(stress_workers[stress_type[j]][i])
        base_url = 'http://%s:8082/v1/cmd' % stress_vm_list[i]
        resp = requests.post(base_url, data=json.dumps(payload))
        if resp.status_code != 200:
            print('failed to start stress at %s' % stress_vm_list[i])
            return START_FAILED
    return SUCCESS


def stress_constant_stop(stress_code, stress_vm=16):
    cmd_name = 'stress'
    stress_vm_list = stress_cpunum_map[stress_vm]
    for i in range(len(stress_vm_list)):
        addr = stress_vm_list[i]
        resp = requests.delete('http://%s:8082/v1/cmd?command=%s&force=true' % (addr, cmd_name))
        if resp.status_code != 200:
            if resp.status_code != 403:
                print('failed to stop stress at %s' % addr)
            elif 'not running' not in resp.text:
                print('failed to stop stress at %s' % addr)

        # resp = requests.delete('http://%s:8081/v1/cmd?command=%s&force=true' % (addr, "sysstat_log"))
        # if resp.status_code != 200:
        #     if resp.status_code != 403:
        #         print('failed to stop logging system stats at %s' % addr)
        #     elif 'not running' not in resp.text:
        #         print('failed to stop logging system stats at %s' % addr)
    
    stressors = stress_code.split(',')
    for i in range(len(stressors)):
        if (stressors[i][0]=='L'):
            requests.delete('http://%s:8090/v1/cmd?command=CAT&force=true' % workload_host_addr)
            base_url = 'http://%s:8090/v1/cmd' % workload_host_addr
            payload = ["CATr", "/home/fsp/czy/modify_cacheways.py", "0x7ff"]
            resp = requests.post(base_url, data=json.dumps(payload))
            #print(resp, json.dumps(payload))
            if resp.status_code != 200:
                print('failed to resume cacheways')
            time.sleep(2)
            requests.delete('http://%s:8090/v1/cmd?command=CATr&force=true' % workload_host_addr)
            break
        elif stressors[i][:3]=='MBW':
            requests.delete('http://%s:8090/v1/cmd?command=CATstress&force=true' % workload_host_addr)
            requests.delete('http://%s:8090/v1/cmd?command=MBW&force=true' % stress_cpunum_map[stress_vm][0])
            
            base_url = 'http://%s:8090/v1/cmd' % stress_cpunum_map[stress_vm][0]
            payload = ["MBWc", "/root/mbw/clean.py"]
            resp = requests.post(base_url, data=json.dumps(payload))
            #print(resp)
            if resp.status_code != 200:
                print('failed to clean mbw')

            base_url = 'http://%s:8090/v1/cmd' % workload_host_addr
            payload = ["CATsr", "/home/fsp/czy/modify_cacheways.py", "0x7ff", "1"]
            resp = requests.post(base_url, data=json.dumps(payload))
            #print(resp, json.dumps(payload))
            if resp.status_code != 200:
                print('failed to resume cacheways')
            
            time.sleep(2)
            requests.delete('http://%s:8090/v1/cmd?command=MBWc&force=true' % stress_cpunum_map[stress_vm][0])
            requests.delete('http://%s:8090/v1/cmd?command=CATsr&force=true' % workload_host_addr)
            break
        elif stressors[i][:3]=='FIO':
            requests.delete('http://%s:8090/v1/cmd?command=FIO&force=true' % stress_cpunum_map[stress_vm][0])
            
            base_url = 'http://%s:8090/v1/cmd' % stress_cpunum_map[stress_vm][0]
            payload = ["FIOc", "/root/clean_fio.py"]
            resp = requests.post(base_url, data=json.dumps(payload))
            if resp.status_code != 200:
                print('failed to clean mbw')
            time.sleep(2)
            requests.delete('http://%s:8090/v1/cmd?command=FIOc&force=true' % stress_cpunum_map[stress_vm][0])
            break
        elif stressors[i][:3]=='NET':
            requests.delete('http://%s:8090/v1/cmd?command=NET&force=true' % stress_cpunum_map[stress_vm][0])
            base_url = 'http://%s:8090/v1/cmd' % stress_cpunum_map[stress_vm][0]
            payload = ["NETc", "/root/clean_iperf3.py"]
            resp = requests.post(base_url, data=json.dumps(payload))
            if resp.status_code != 200:
                print('failed to clean iperf3')
            time.sleep(2)
            requests.delete('http://%s:8090/v1/cmd?command=NETc&force=true' % stress_cpunum_map[stress_vm][0])
            os.system("pkill iperf3")
            break
        elif stressors[i][:2]=='CU':
            requests.delete('http://%s:8090/v1/cmd?command=CU&force=true' % workload_host_addr)
            
            base_url = 'http://%s:8090/v1/cmd' % workload_host_addr
            payload = ["CUr", "/home/fsp/czy/ch_cpu_util/ch_cpu.py"]
            resp = requests.post(base_url, data=json.dumps(payload))
            #print(resp)
            if resp.status_code != 200:
                print('failed to remove cpu limit')
            time.sleep(2)
            requests.delete('http://%s:8090/v1/cmd?command=CUr&force=true' % workload_host_addr)
            break

def stress_all():
    current_run_idx = 0
    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    wf = open('%s-stress.log' % current, 'w')
    #url = 'http://%s:8090/v1/cmd' % stress_cpunum_map[16][0]
    name = 'stress'
    retry_stop = False
    already_run = False
    last_duration = -1
    time.sleep(15)
    print("stress start at", time.strftime("%Y %m %d %H %M %S", time.localtime(time.time())))
    for stress_type in stress_type_map:
        for stress_level in stress_level_map[stress_type]:
            stress_code = stress_type + str(stress_level)
            stress_status = stress_constant_start(stress_code)
            print(stress_code, stress_status)
            duration = _wait_and_record_stress(wf, stress_type, stress_level)
            stress_constant_stop(stress_code)
            time.sleep(1)
    print("stress finish at", time.strftime("%Y %m %d %H %M %S", time.localtime(time.time())))
    wf.flush()
    wf.close()


def redis_run(loadlevel='2'):
    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    #print(time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
    num = 12000000+min(int(loadlevel),12)*5000000
    with open('redis-result.log', 'w') as f:
        p = subprocess.Popen(
            ['/root/redis-benchmark', '-h', '192.168.0.233', '-n', str(num),
             '-c', loadlevel, '--dfile', '-t', 'set'], stderr=f) #2e OK, 4e crashed
        p.wait()
    #print(time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
    os.rename('result-dense.csv', '%s-redis-result.csv' % current)


def ab_run():
    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    with open('%s-ab-result.log' % current, 'w') as f:
        # change 40000 to 80000
        p = subprocess.Popen(['/root/ab', '-l', '-n', '11500', '-c', '1',#240000->6315
                              'http://192.168.0.233/'],#128
                             env={'LD_LIBRARY_PATH': '/root'}, stderr=f)
        p.wait()


def sysbench_run(loadlevel='1'):
#    p = subprocess.Popen([
#            '/root/sysbench', '--num-threads=1', '--max-time=180',
#            '--test=oltp', '--db-driver=mysql', '--mysql-host=192.168.0.233',
#            '--mysql-port=3306', '--mysql-user=sysbench',
#            '--mysql-password=sysbench', '--report-interval=1', 'prepare'],
#            )
#    p.wait()
    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    with open('%s-sysbench-result.log' % current, 'w') as f:
        p = subprocess.Popen([
            '/root/sysbench', '--num-threads=%s'%loadlevel, '--max-requests=4500000', '--max-time=600',
            '--test=oltp', '--db-driver=mysql', '--mysql-host=192.168.0.233',
            '--mysql-port=3306', '--mysql-user=sysbench',
            '--mysql-password=sysbench', '--report-interval=1', 'run'],
            stdout=f)
        p.wait()


def _build_url(addr, name=None):
    if name:
        return 'http://%s:8081/v1/cmd?command=%s' % (addr, name)
    else:
        return 'http://%s:8081/v1/cmd' % addr


def ffmpeg_run():
    _app = 'run-ffmpeg'
    addr = '192.168.0.233'
    req_body = [_app, '/root/%s.sh' % _app]
    res = requests.post(_build_url(addr), data=json.dumps(req_body))
    if res.status_code != 200:
        print('failed to start workload deployment request, '
              'code: %d' % res.status_code)
        return

    while True:
        time.sleep(10)
        res = requests.get(_build_url(addr, _app))
        if res.status_code != 200:
            print('failed to query workload deployment status, '
                  'code: %d' % res.status_code)
            continue

        tokens = res.text.strip().split(',')
        if tokens[0] == 'false':
            continue

        res = requests.delete(_build_url(addr, _app))
        if res.status_code != 200:
            print('failed to stop workload deployment request, '
                  'code: %d' % res.status_code)
        break


def rabbitmq_run(loadlevel='1'):
    java_home = '/root/software/jdk1.8.0_212'
    cmd_path = '/root/software/rabbitmq-perf-test/bin/runjava'
    n_producer, n_comsumer = 1, int(loadlevel)
    duration = 600 # second
    message_size = 1024 # byte
    addr = 'amqp://192.168.0.233'

    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    with open('%s-rabbitmq-result.log' % current, 'w') as f:
        p = subprocess.Popen([cmd_path, 'com.rabbitmq.perf.PerfTest', 
                              '-x', str(n_producer), 
                              '-y', str(n_comsumer),
                              '-z', str(duration),
                              '-s', str(message_size),
                              '-uri', addr],
                             env={'JAVA_HOME': java_home},
                             stdout=f)
        p.wait()


def etcd_run(loadlevel='1'):
    cmd_path = '/root/software/etcd-benchmark'
    endpoints = 'http://192.168.0.233:2379'
    total = min(int(loadlevel), 6) * 1000000#288000 # 11200000 # put request count
    key_size = 8
    value_size = 8

    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    #print("start at: ", time.strftime("%Y %m %d %H %M %S", time.localtime(time.time())))
    with open('%s-etcd-result.log' % current, 'w') as f:
        p = subprocess.Popen([cmd_path,
                              '--endpoints', endpoints,
                              '--clients', loadlevel,
                              'put',
                              '--total', str(total),
                              '--key-size', str(key_size),
                              '--val-size', str(value_size)],
                              stdout=f)
        p.wait()
    #print("finish at: ", time.strftime("%Y %m %d %H %M %S", time.localtime(time.time())))


def kafka_run(loadlevel='-1'):
#    kafka_cold_run() 
    java_home = '/root/software/jdk1.8.0_212'
    kafka_path = '/root/software/kafka'
    cmd_path = os.path.join(kafka_path, 'bin/kafka-producer-perf-test.sh')
    topic = 'topicname' # anything is ok, new topic will be create.
    num_records = 20800000
    if loadlevel!='-1':
        num_records = 650*int(loadlevel) #max 32000
    record_size = 4096
    addr = '192.168.0.233'
    producer_props = 'bootstrap.servers=%s:9092' % addr

    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    with open('%s-kafka-result.log' % current, 'w') as f:
        p = subprocess.Popen([cmd_path,
                              '--throughput', loadlevel, # -1 mean no control client maximum throughput
                              '--topic', topic,
                              '--num-records', str(num_records),
                              '--record-size', str(record_size),
                              '--producer-props', producer_props],
                              env={'JAVA_HOME': java_home},
                              stdout=f)
        p.wait()


def mongodb_run(loadlevel='5'):
    java_home = '/root/software/jdk1.8.0_212' # change the value based on the site requirements.
    ycsb_path = '/root/software/ycsb/bin/ycsb'
    workload_path = '/root/mongodb-workload-10min'
    op_count = 137000000

    with open('/dev/null', 'w') as devnull:
        p = subprocess.Popen([ycsb_path,
                          'load', 'mongodb',
                          '-P', workload_path],stdout=devnull)
        p.wait()

    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    with open('%s-mongoDB-result.log' % current, 'w') as f:
        with open('/dev/null', 'w') as devnull:
            p = subprocess.Popen([ycsb_path,
                                  'run', 'mongodb',
                                  '-P', workload_path,
                                  '-threads', loadlevel,
                                  '-s', # mean print status to stderr
                                  '-p', 'operationcount=%d' % op_count,
                                  '-jvm-args', '-Xms6g -Xmx6g'
                                 ],
                                 env={'JAVA_HOME': java_home},
                                 stdout=devnull, # avoid too many useless output in terminal
                                 stderr=f)
            p.wait()


def cassandra_run(loadlevel='100'):
    java_home = '/root/software/jdk1.8.0_212' # change the value based on the site requirements.
    ycsb_path = '/root/software/ycsb/bin/ycsb'
    workload_path = '/root/cassandra-workload-10min'
    op_count = 72000000
    addr = '192.168.0.233'
    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

    with open('/dev/null', 'w') as devnull:
        p = subprocess.Popen([ycsb_path,
                          'load', 'cassandra-cql',
                          '-P', workload_path],stdout=devnull)
        p.wait()

    with open('%s-cassandra-result.log' % current, 'w') as f:
        with open('/dev/null', 'w') as devnull:
            p = subprocess.Popen([ycsb_path,
                                  'run', 'cassandra-cql',
                                  '-threads', loadlevel,
                                  '-P', workload_path,
                                  '-s', # mean print status to stderr
                                  '-p', 'operationcount=%d' % op_count,
                                  '-p', 'hosts=%s' % addr,
                                  "-jvm-args", "-Xms6g -Xmx6g",
                                  ],
                                  env={'JAVA_HOME': java_home},
                                  stdout=devnull, # avoid too many useless output in terminal
                                  stderr=f,)
            p.wait()


def hbase_run(loadlevel='10'): # <=100
    java_home = '/root/software/jdk1.8.0_212' # change the value based on the site requirements.
    ycsb_path = '/root/software/ycsb-hbase2-binding/bin/ycsb'
    workload_path = '/root/hbase-workload-10min'
    conf_path = '/root/software/hbase/conf'
    op_count = 81000000
    addr = '192.168.0.233'

#    p = subprocess.Popen([ycsb_path,
#                          'load', 'hbase2',
#                          '-cp', '/root/software/hbase/conf'
#                          '-P', workload_path])
#    p.wait()

    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    with open('%s-hbase-result.log' % current, 'w') as f:
        with open('/dev/null', 'w') as devnull:
            p = subprocess.Popen([ycsb_path,
                                  'run', 'hbase2',
                                  '-threads', loadlevel,
                                  '-P', workload_path,
                                  '-s', # mean print status to stderr
                                  '-p', 'operationcount=%d' % op_count,
                                  '-cp', conf_path,
                                  '-p', 'table=usertable',
                                  '-p', 'columnfamily=family',
                                  "-jvm-args", "-Xms6g -Xmx6g",
                                  ],
                                  env={'JAVA_HOME': java_home},
                                  stdout=devnull, # avoid too many useless output in terminal
                                  stderr=f,)
            p.wait()


def spark_run():
    java_home = '/root/software/jdk1.8.0_212'
    addr = '192.168.0.32'
    spark_home = '/root/software/spark-2.4.7-bin-hadoop2.7'
    count = 500
    dataset_path = '/root/software/web-Stanford.txt'

    spark_submit = '%s/bin/spark-submit' % spark_home
    master_addr = 'spark://%s:7077' % addr
    class_name = 'org.apache.spark.examples.SparkPageRank'
    jar_path = '%s/examples/jars/spark-examples_2.11-2.4.7.jar' % spark_home


    current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    with open('%s-spark-result.log' % current, 'w') as f:
        with open('/dev/null') as devnull:
            p = subprocess.Popen([spark_submit,
                                  '--master', master_addr,
                                  '--class', class_name,
                                  '--conf', 'spark.driver.extraJavaOptions="-Xss60M"',
                                  jar_path,
                                  'file://' + dataset_path,
                                  str(count),],
                                  env={'JAVA_HOME': java_home},
                                  stderr=devnull, stdout=devnull)
            p.wait()
            current = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
            f.write(current)

def cpu2006_run(job, num):
    _app = 'run-cpu2006'
    addr = '192.168.0.233'
    req_body = [_app, '/root/%s.py' % _app, job, num]
    res = requests.post('http://%s:8090/v1/cmd' % addr, data=json.dumps(req_body))
    if res.status_code != 200:
        print('failed to start workload deployment request, '
              'code: %d' % res.status_code)
        return

    while True:
        time.sleep(10)
        res = requests.get('http://%s:8090/v1/cmd?command=%s' % (addr,_app))
        if res.status_code != 200:
            print('failed to query workload deployment status, '
                  'code: %d' % res.status_code)
            continue

        tokens = res.text.strip().split(',')
        if tokens[0] == 'false':
            continue

        res = requests.delete('http://%s:8090/v1/cmd?command=%s' % (addr,_app))
        if res.status_code != 200:
            print('failed to stop workload deployment request, '
                  'code: %d' % res.status_code)
        break 

app_run_map = {
    'ab': ab_run,
    'sysbench': sysbench_run,
    'redis': redis_run,
    'ffmpeg': ffmpeg_run,
    'rabbitmq': rabbitmq_run,
    'etcd': etcd_run,
    'kafka': kafka_run,
    'mongodb': mongodb_run,
    'cassandra': cassandra_run,
    'hbase': hbase_run,
    'spark': spark_run
}


if __name__ == '__main__':
    app = sys.argv[1]
    more_stress = False
    stress_once_idx = None
    process = multiprocessing.Process(target=stress_all)
    process.start()
    print("app start at: ", time.strftime("%Y %m %d %H %M %S", time.localtime(time.time())))
    if 'cpu2006' in app:
        args = app.split('-')
        if len(args) == 2:
            cpu2006_run(args[1],'4')
        else:
            cpu2006_run(args[1],args[2])
    else:
        if len(sys.argv)<3 or sys.argv[2]=='0':
            app_run_map[app]()
        else:
            app_run_map[app](sys.argv[2])
    print("app finish at: ", time.strftime("%Y %m %d %H %M %S", time.localtime(time.time())))
    process.join()
#    if app == 'ffmpeg':
#        more_stress = True
        # stress_once_idx = random.randint(1, 5)
#    stress_code = sys.argv[2]
#    queue = multiprocessing.Queue(1)
#    process = multiprocessing.Process(
#        target=stress_loop, args=(queue, stress_code, more_stress, stress_once_idx))
#    process.start()
    # if 'stress' in app:
    #     stress_vm=int(app[7:])
    #     stress_constant_start(stress_code, stress_vm, sys.argv[3])  
    # else:
    #     stress_vm = 16
    #     stress_constant_start(stress_code, stress_vm)  
    # if 'cpu2006' in app:
    #     args = app.split('-')
    #     if len(args) == 2:
    #         cpu2006_run(args[1],'4')
    #     else:
    #         cpu2006_run(args[1],args[2])
    # elif 'stress' in app:
    #     time.sleep(int(sys.argv[3]))
    # else:
    #     app_run_map[app]()
    # stress_constant_stop(stress_code, stress_vm)
#    queue.put_nowait(True)
#    process.join()
