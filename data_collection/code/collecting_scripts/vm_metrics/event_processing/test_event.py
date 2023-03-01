# %%
# test which event can be used in a physical machine
# if a event value = 0, then it is deleted from event pool

import subprocess
import time

codes = []
with open("./total_code", "r") as f:
    for line in f.readlines():
        codes.append(line.strip('\n'))


f = open("total_code_selected", "w")

for code in codes:
    perf_cmd = "perf stat -e %s ./showevtinfo cycles" %(code)
    print("Command: %s" %(perf_cmd))
    pPerf = None
    rc = None
    pPerf = subprocess.Popen([perf_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    while True:
        if pPerf.poll() == None:
            output = pPerf.communicate()[1]
            time.sleep(1)
        else:
            break
    for line in output.split('\n'):
        linestrip = line.strip().split()
        # print(linestrip)
        if len(linestrip) == 2:
            if linestrip[0] != '0':
                f.writelines("%s\n" %(code))
                print(code, linestrip[0])
            else:
                print("Value = 0 , remove from codes")
            break

f.close()