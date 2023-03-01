#!/usr/bin/env python
import os
import re
import sys
import subprocess
import signal

workers = int(sys.argv[1])
if len(sys.argv) > 2:
    mem_cap = sys.argv[2]
else:
    mem_cap = "256"

proclist = []
for i in range(workers):
    command='taskset -c %d /root/mbw/mbw %s'% (i, mem_cap)
    proc=subprocess.Popen(['bash','-c',command],preexec_fn=os.setsid)
    proclist.append(proc)

def proc_clear(proclist):
    for proc in proclist:
        proc.terminate()
        proc.wait()

try:
    while(True):
        pass
finally:
    proc_clear(proclist)
