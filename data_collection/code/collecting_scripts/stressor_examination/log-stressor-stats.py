import subprocess
import sys
import time
import datetime
from multiprocessing import Pool


def log_top(stressor,duration):
    t = datetime.datetime.now()
    x = t.strftime('%Y%m%d%H%M%S')
    args = ["top -b -n 1 | awk 'NR<=100'"]
    with open('%s-top-%s.log' % (x,stressor),'w') as f:
        tsc = 0
        timeout = int(duration)
        while tsc < timeout:
            tsc += 1
            p = subprocess.Popen(args, stdout=f, shell=True)
            p.wait()
            time.sleep(1)

def log_mpstat(stressor,duration):
    t = datetime.datetime.now()
    x = t.strftime('%Y%m%d%H%M%S')
    args = ["mpstat", "-A", "1", "%s" % duration]
    with open('%s-mpstat-%s.log' % (x,stressor),'w') as f:
        p = subprocess.Popen(args, stdout=f)
        p.wait()

def log_vmstat(stressor,duration):
    t = datetime.datetime.now()
    x = t.strftime('%Y%m%d%H%M%S')
    args = ["vmstat", "1", "%s" % duration]
    with open('%s-vmstat-%s.log' % (x,stressor),'w') as f:
        p = subprocess.Popen(args, stdout=f)
        p.wait()

def log_iostat(stressor,duration):
    t = datetime.datetime.now()
    x = t.strftime('%Y%m%d%H%M%S')
    args = ["iostat", "-d", "-x", "-k", "-p", "ALL", "1", "%s" % duration]
    with open('%s-iostat-%s.log' % (x,stressor),'w') as f:
        p = subprocess.Popen(args, stdout=f)
        p.wait()

def log_softirq(stressor,duration):
    t = datetime.datetime.now()
    x = t.strftime('%Y%m%d%H%M%S')
    args = ["cat", "/proc/softirqs"]
    with open('%s-softirq-%s.log' % (x,stressor),'w') as f:
        tsc = 0
        timeout = int(duration)
        while tsc < timeout:
            tsc += 1
            p = subprocess.Popen(args, stdout=f)
            p.wait()
            time.sleep(1)

def log_netstat(stressor,duration):
    t = datetime.datetime.now()
    x = t.strftime('%Y%m%d%H%M%S')
    args = ["netstat -s | awk 'NR<=8 || NR>=28 && NR<=39'"]
    with open('%s-netstat-%s.log' % (x,stressor),'w') as f:
        tsc = 0
        timeout = int(duration)
        while tsc < timeout:
            tsc += 1
            p = subprocess.Popen(args, stdout=f, shell=True)
            p.wait()
            time.sleep(1)

def log_sar_net(stressor,duration):
    t = datetime.datetime.now()
    x = t.strftime('%Y%m%d%H%M%S')
    args = ["sar", "-n", "DEV,EDEV,SOCK,IP,EIP,TCP,ETCP", "1", "%s" % duration]
    with open('%s-sar_net-%s.log' % (x,stressor),'w') as f:
        p = subprocess.Popen(args, stdout=f)
        p.wait()

run_app = [log_top, log_mpstat, log_vmstat, log_iostat, log_softirq, log_netstat, log_sar_net]

if __name__ == '__main__':
    worker_num=4
    pool = Pool(processes=worker_num)
    for i in range(3):
        pool.apply_async(run_app[i], args=(sys.argv[1], sys.argv[2]))
#    pool.apply_async(run_app[3], args=(sys.argv[1], sys.argv[2]))
    pool.apply_async(run_app[4], args=(sys.argv[1], sys.argv[2]))
#    pool.apply_async(run_app[5], args=(sys.argv[1], sys.argv[2]))
#    pool.apply_async(run_app[6], args=(sys.argv[1], sys.argv[2]))
    pool.close()
    pool.join()
