#!/usr/bin/env python
import os
import re
import sys
import psutil
import rtmonitor
import subprocess
import cat

coresmap=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
sizemap=[80]*len(coresmap)
PHY_MBW=140
def bw_consume(level,corelist):
    cpu_count=psutil.cpu_count()
    phy_count=int(cpu_count/2)
    pairs=[(i,i+phy_count) for i in range(phy_count)]
    for coreid in corelist:
        for pair in pairs:
            if coreid in pair:
                pairs.remove(pair)
    bw_cores=[]
    corenum=coresmap[level]
    for i in range(int(corenum/2)):
        if len(pairs)>0:
            p=pairs.pop(0)
            bw_cores.append(p[0])
            bw_cores.append(p[1])
    if corenum%2 and len(pairs)>0:
        p=pairs.pop(0)
        bw_cores.append(p[0])
    #limit the LLC cache used by the bw bubble
    #control the influence to cache caused by bw bubble
    cat.limitcache(bw_cores,3,1)
    #start the bw bubble
    proclist=[]
    for core in bw_cores:
        command='taskset -c %d ../bin/bw %d '%(core,sizemap[level]*(10**6))
        proc=subprocess.Popen(['bash','-c',command],preexec_fn=os.setsid)
        proclist.append(proc)
    return proclist
def bw_clear(proclist):
    cat.clearcat()
    for proc in proclist:
        proc.terminate()
        proc.wait()
def shrink(cores):
    cat.clearcat()
    #original mem bandwidth occupution
    _mbw=rtmonitor.mbw_occupy(cores)
    trace={}
    trace[0]=_mbw
    for level in range(1,len(coresmap)):
        procs=bw_consume(level,cores)
        mbw=rtmonitor.mbw_occupy(cores)
        trace[level]=mbw
        bw_clear(procs)
        if mbw<=_mbw*0.9:
            break
    return trace

def quantify(corelist):
    trace=shrink(corelist)
    cat.clearcat()
    #print(trace)
    #pqos return the bandwidth in MB/s , first change it into GB/s
    #20 pressure levels in total
    press=20*trace[0]/1024.0/PHY_MBW
    sensitivity=len(coresmap)-len(trace)
    return press,sensitivity,trace

if __name__=='__main__':
    cores=6
    phycores=int(psutil.cpu_count()/2)
    tmplist=list(range(0,cores))+list(range(phycores,phycores+cores))
    quantify(tmplist)
