#!/usr/bin/env python
import os
import sys
import re
import time
import subprocess
import threading
import cgmonitor
import pfmonitor
from statistics import mean
PERF_INTERVAL=4

def cache_occupy(corelist):
    occupyfile='/dev/shm/cache_occpy'
    corelist=[str(x) for x in corelist]
    strcores=','.join(corelist)
    if os.path.exists(occupyfile):
        os.system('rm -f %s'%occupyfile)
    command='pqos -m "llc:[%s]" -t 5 -o %s'%(strcores,occupyfile)
    os.system(command)
    with open(occupyfile,'r') as ocf:
        lines=ocf.readlines()
        line=lines[-1]
        line=re.split(' ',line)
        line=list(filter(lambda x:len(x)>0,line))
        oc=float(line[3])
        return oc

def mbw_occupy(corelist):
    #return the consumed memory bandwidth
    occupyfile='/dev/shm/mbw_occupy'
    corelist=[str(x) for x in corelist]
    strcores=','.join(corelist)
    if os.path.exists(occupyfile):
        os.system('rm -f %s'%occupyfile)
    command='pqos -m "all:[%s]" -t 5 -o %s'%(strcores,occupyfile)
    os.system(command)
    with open(occupyfile,'r') as ocf:
        lines=ocf.readlines()
        line=lines[-1]
        line=re.split(' ',line)
        line=list(filter(lambda x:len(x)>0,line))
        #return MBR+MBL
        return float(line[4])+float(line[5])

def get_interface_bytes(rw,interface):
    with open('/sys/class/net/'+interface+'/statistics/'+rw+'_bytes','r') as f:
        return int(f.read())

def nbw_occupy_interface(interface='eth0'):
    #return the consumed network bandwidth of specified interface
    #cannot get the network bandwidth at the cg-level
    tx1=get_interface_bytes('tx',interface)
    rx1=get_interface_bytes('rx',interface)
    time.sleep(5)
    tx2=get_interface_bytes('tx',interface)
    rx2=get_interface_bytes('rx',interface)
    tx_speed=round((tx2-tx1)/1024.0/1024.0/1024.0/5.0,4)
    rx_speed=round((rx2-rx1)/1024.0/1024.0/1024.0/5.0,4)
    #in Gbit/s
    return tx_speed*8.0,rx_speed*8.0

def get_port_bytes(port):
    process=subprocess.Popen(['iptables','-L','-v','-n','-x'],
                             stdout=subprocess.PIPE)
    out,err=process.communicate()
    out=re.split('\n',out.decode())
    pattern=':{}'.format(port)
    out=list(filter(lambda x:pattern in x,out))
    out=list(filter(lambda x:len(x)>0,re.split(' ',out[0])))
    return int(out[1])

def nbw_occupy(port):
    #Add rules
    os.system('iptables -A OUTPUT -p tcp --sport {}'.format(port))
    WAIT=5
    rx1=0
    tx1=get_port_bytes(port)
    time.sleep(WAIT)
    rx2=0
    tx2=get_port_bytes(port)
    rx_speed=(rx2-rx1)/WAIT/1024.0/1024.0/1024.0
    tx_speed=(tx2-tx1)/WAIT/1024.0/1024.0/1024.0
    #remove the rules
    os.system('iptables -D OUTPUT -p tcp --sport {}'.format(port))
    #to Gbit/s
    return tx_speed*8.0,rx_speed*8.0


def get_blk_iops(cgpath):
    iops_f=os.path.join(cgpath,'blkio.throttle.io_serviced')
    with open(iops_f,'r') as iof:
        content=iof.readlines()
        riops=0
        wiops=0
        for line in content:
            if 'Read' in line:
                line=re.split(' ',line)
                riops+=int(line[2])
            elif 'Write' in line:
                line=re.split(' ',line)
                wiops+=int(line[2])
        return [riops,wiops]

def get_blk_bytes(cgpath):
    byte_f=os.path.join(cgpath,'blkio.throttle.io_service_bytes')
    with open(byte_f,'r') as bf:
        content=bf.readlines()
        Read=0
        Write=0
        for line in content:
            if 'Read' in line:
                line=re.split(' ',line)
                Read+=int(line[2])
            elif 'Write' in line:
                line=re.split(' ',line)
                Write+=int(line[2])
        return [Read,Write]

def dbw_occupy(cgname):
    BLK_PREFIX='/sys/fs/cgroup/blkio'
    BLK_WAIT=4
    cgpath= os.path.join(BLK_PREFIX,cgname)
    if not os.path.exists(cgpath):
        return 0.0,0.0
    bytes1=get_blk_bytes(cgpath)
    iops1=get_blk_iops(cgpath)
    time.sleep(BLK_WAIT)
    bytes2=get_blk_bytes(cgpath)
    iops2=get_blk_iops(cgpath)
    for i in range(2):
        bytes2[i]=(bytes2[i]-bytes1[i])/ BLK_WAIT/1024.0/1024.0
        iops2[i]=(iops2[i]-iops1[i])/BLK_WAIT
    # IOPS(R+W) , Bandwidth(R+W)
    return sum(iops2),sum(bytes2)

def s2f(s):
    value=0.0
    try:
        value=float(s.replace(',',''))
    except ValueError as ve:
        print(ve)
    return value

def load_kcpsfile(fpath):
    f=open(fpath,'r')
    lines=f.readlines()
    lines=list(filter(lambda x:x[0]!='#',lines))
    lines=list(filter(lambda x:len(x)>2,lines))
    f.close()
    data={}
    data['instructions']=[]
    data['cache-misses']=[]
    for line in lines[:-2]:
        #drop the last second because
        #somtime perf write the last second two times
        line=line.strip()
        line=re.split(' ',line)
        line=list(filter(lambda x:len(x)>0,line))
        time=float(line[0])
        value=s2f(line[1])
        index=line[2]
        data[index].append((time,value))
    kcps=[]
    mpki=[]
    misses=data['cache-misses']
    instru=data['instructions']
    for i in range(1,len(misses)):
        dur=misses[i][0]-misses[i-1][0]
        kcps.append(misses[i][1]/dur/1000.0)
        mpki.append(1000*misses[i][1]/instru[i][1])
    return mean(kcps),mean(mpki)

def kcps_monitor(corelist):
    kcpsfile='/dev/shm/kcps_monitor'
    corelist=[str(x) for x in corelist]
    strcores=','.join(corelist)
    if os.path.exists(kcpsfile):
        os.system('rm -f %s'%kcpsfile)

    command='sudo perf stat -e instructions,cache-misses -C %s -I 1000 -o %s sleep 5'%(strcores,kcpsfile)
    os.system(command)
    data=load_kcpsfile(kcpsfile)
    return data[0]


def monitor_cg(cgname,path,timelength):
    global PERF_INTERVAL
    if not os.path.isdir(path):
        os.mkdir(path)
    terminate=threading.Event()
    cgm=threading.Thread(target=cgmonitor.checkpoint,
                            args=(cgname,path,1,terminate))
    cgm.start()
    pfoutfile=os.path.join(path,'perf.txt')
    pfcommand=pfmonitor.getcommand(cgname,PERF_INTERVAL)
    pf=open(pfoutfile,'w')
    pf.write('#'+str(time.time())+'\n')
    pfm=subprocess.Popen(['bash','-c',pfcommand],
                            preexec_fn=os.setsid,stderr=pf)
    pf.close()
    #wait a while
    #collect the system indexes
    time.sleep(timelength)
    #stop the montoring threads
    terminate.set()
    cgm.join()
    pfm.terminate()
    pfm.wait()
