#!/usr/bin/env python
import os
import re
import sys
import time
import json
import psutil
import rtmonitor
import cat
import numpy as np

STANDARD_RULER=20
CACHE_WAYS=cat.cache_ways()
CACHE_SIZE=25600
SHRINK_STEP=4
KCPS_RULER=30*10**4
def shrink(cores):
    global CACHE_SIZE
    global CACHE_WAYS
    global SHRINK_STEP
    cat.clearcat()
    time.sleep(SHRINK_STEP)
    _kcps=rtmonitor.kcps_monitor(cores)
    trace={}
    trace[CACHE_WAYS]=_kcps
    for ways in range(CACHE_WAYS-1,0,-1):
        time.sleep(SHRINK_STEP)
        mask=(1<<ways)-1
        cat.limitcache(cores,2,mask)
        kcps=rtmonitor.kcps_monitor(cores)
        trace[ways]=kcps
        if kcps>_kcps*1.10:
            print('Stop at %d ways'%ways)
            break
    return trace

def distance(ta,tb):
    num=0
    dissum=0
    for key in ta:
        if key in tb:
            num+=1
            dissum+=(ta[key]-tb[key])**2
    if num:
        return np.sqrt(dissum/num)
    else:
        return 10**20
def search(strace,traces):
    #find the most similar one
    #strace is the performance shrink trace of the
    #target application, traces is the performance
    #traces of the
    #traces:{'cores size':{ways:kcps} }
    sortlist=[]
    dis=10**20
    re=None
    for trace in traces:
        d=distance(traces[trace],strace)
        #print(trace,d)
        if d<dis:
            dis=d
            re=trace
        sortlist.append((d,trace))

    sortlist.sort()
    #print(sortlist)
    return re

def load_traces(path='../data/cache.json'):
    traces={}
    tmpdict={}
    with open(path,'r') as jf:
        tmpdict=json.load(jf)
    for plevel in tmpdict:
        trace=tmpdict[plevel]
        plevel=int(plevel)
        trace=dict([(int(key),trace[key]) for key in trace])
        traces[plevel]=trace
    return traces

def quantify(corelist):
    traces=load_traces()
    strace=shrink(corelist)
    nearest=search(strace,traces)
    #clear the cat configuration after quantification
    cat.clearcat()
    #print(strace)
    #print(nearest)
    # 1<=sensitivity<=CACHE_WAYS
    #normalize to the standard:20
    sensitivity=(CACHE_WAYS-len(strace)+1)*STANDARD_RULER/ CACHE_WAYS
    ways=max(strace.keys())
    #press=int(100*strace[ways]/KCPS_RULER)
    #return press,sensitivity
    return int(nearest),sensitivity,strace

def test():
    avg=[]
    traces=load_traces()
    for conf in traces:
        values=list(traces[conf].values())
        avg_kcps=sum(values)/float(len(values))
        avg.append((avg_kcps,conf))
    avg.sort()
    for a in avg:
        print(a)

if __name__ =='__main__':
    #test()
    cores=4
    phycores=int(psutil.cpu_count()/2)
    tmplist=list(range(0,cores))+list(range(phycores,phycores+cores))
    quantify(tmplist)
