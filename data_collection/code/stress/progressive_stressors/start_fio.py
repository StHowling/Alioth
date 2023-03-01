#!/usr/bin/env python
import os
import re
import sys
import subprocess
import signal

iodepths = int(sys.argv[1])

command='fio -name=mytest -filename=/test/test.img -direct=1 -iodepth=%s -thread -rw=randrw -ioengine=libaio -bs=16k -size=5G -numjobs=2 -runtime=300 -group_reporting'% (iodepths)
#proc=subprocess.Popen(['bash','-c',command])
#proc.wait()
os.system(command)
