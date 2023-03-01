#!/usr/bin/env python
import os
import sys
import re
import subprocess

def clearcat():
    command='pqos -R'
    os.system(command)

def limitcache(corelist,cosid,ways):
    if len(corelist)==0:
        return
    corelist=[str(x) for x in corelist]
    cores=','.join(corelist)
    #allocated only one ways
    #used for the bubble banwidth
    os.system('pqos -e \"llc:{}={}\" '.format(cosid,hex(ways)))
    os.system('pqos -a \"llc:{}={}\" '.format(cosid,cores))

def limitcache_cg(cgname,cosid,ways):
    if not os.path.exists('/sys/fs/cgroup/cpuset/%s'%cgname):
        return None
    f=open('/sys/fs/cgroup/cpuset/%s/cpuset.cpus'%cgname,'r')
    cores=f.read().strip()
    f.close()
    os.system('pqos -e \"llc:{}={}\" '.format(cosid,hex(ways)))
    os.system('pqos -a \"llc:{}={}\" '.format(cosid,cores))

def cache_occupy(corelist):
    occupyfile='/dev/shm/cache_occpy'
    corelist=[str(x) for x in corelist]
    strcores=','.join()
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
    occupyfile='/dev/shm/mbm_occupy'
    corelist=[str(x) for x in corelist]
    strcores=','.join()
    if os.path.exists(occupyfile):
        os.system('rm -f %s'%occupyfile)
    command='pqos -m "all:[%s]" -t t -o %s'%(strcores,occupyfile)
    os.system(command)
    with open(occupyfile,'r') as ocf:
        lines=ocf.readlines()
        line=lines[-1]
        line=re.split(' ',line)
        line=list(filter(lambda x:len(x)>0,line))
        #return MBR+MBL
        return float(line[4])+float(line[5])

def cache_ways():
    process=subprocess.Popen(['pqos','-s'],stdout=subprocess.PIPE)
    out,err=process.communicate()
    out=re.split('\n',out.decode())
    out=list(filter(lambda x:"L3CA COS0" in x,out))
    pattern='0x[137f]+'
    ways=re.search(pattern,out[0])
    ways=ways.group(0)[2:]
    tmpmap={'1':1,'3':2,'7':3,'f':4}
    sum=0
    for i in range(len(ways)):
        sum+=tmpmap[ways[i]]
    return sum
