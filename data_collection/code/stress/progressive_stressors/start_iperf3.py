#!/usr/bin/env python
import os
import re
import sys
import subprocess
import signal

bandwidth = int(sys.argv[1])*100
bandwidth = str(bandwidth)+'M'
#nd = "iperf3 -c 192.168.0.180 -u -b %sM -t 10000" % (bandwidth)
proc=subprocess.Popen(['iperf3','-c','192.168.0.180','-u','-b',bandwidth,'-t','10000'])
print('connected')
proc.wait()
#os.system(command)
