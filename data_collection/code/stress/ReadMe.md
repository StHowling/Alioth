# intel-cmt-cat python 

## Caution!
Most of the cpu in our test environment are intel xeon 6151. 
It has the following abilities. 
```
INFO: Monitoring capability detected
INFO: CPUID.0x7.0: L3 CAT supported
INFO: L3 CDP is disabled
INFO: L3CA capability detected
INFO: L3 CAT details: CDP support=1, CDP on=0, #COS=16, #ways=11, ways contention bit-mask 0x600
INFO: L3 CAT details: cache size 25952256 bytes, way size 2359296 bytes
INFO: CPUID 0x10.0: L2 CAT not supported!
INFO: L2CA capability not detected
INFO: Detected MBA version 1.0
INFO: Detected Per-Core MBA controls
INFO: MBA capability detected
INFO: MBA details: #COS=8, linear, max=90, step=10
DEBUG: allocation init OK
DEBUG: Max RMID per monitoring cluster is 144
INFO: Detected perf monitoring support for LLC Occupancy
INFO: Detected perf monitoring support for Local Memory B/W
INFO: Detected perf monitoring support for Total Memory B/W
INFO: Detected perf monitoring support for Remote Memory B/W
INFO: Detected perf monitoring support for Retired CPU Instructions
INFO: Detected perf monitoring support for Unhalted CPU Cycles
INFO: Detected perf monitoring support for Instructions/Cycle
INFO: Detected perf monitoring support for LLC Misses
```
It only support **_L3 CAT and MBA_**. 
L3 has **11 ways** and it has **16** L3 CAT classes of service (CLOS). 

The kernel version of our environment is `3.10.0-862.14.1.6_48.x86_64`. 
So our environment does not have `perf` and `resctrl` extension support, 
which means you should use command `pqos` without `-I`. 
pqos will programs the techonologies via **Model Specific Register(MSR)**.

Before using any of the command below (`pqos` or `python`), 
you should install the pqos binary or pqos python package.

`pqos` binary:
```
cd intel-cmt-cat-4.1.0
make
sudo make install
```

`pqos` python package:
```
cd intel-cmt-cat-4.1.0/lib/python
# create your python virtual environment via virtualenv or conda
python setup.py build
python setup.py install
# activate the environment before you use python
```

## List all of the cores and CLOS
Use `pqos -s` to see all the cores and CLOS. 
If you only get part of the cores, 
you need to change your `cgroup` with these command. 
```
# if you don't have test cgroup, please create one!
mkdir /sys/fs/cgroup/cpuset/test
echo "ALL_OF_YOUR_CORE (typically 0-71)" > /sys/fs/cgroup/cpuset/test/cpuset.cpus
echo "ALL_OF_YOUR_NUMA_NODE (typically 0-1)" > /sys/fs/cgroup/cpuset/test/cpuset.mems
# this is the actual command to add your shell process to a cgroup
echo $$ >> /sys/fs/cgroup/cpuset/test/tasks
```

## L3 Cache Allocation Technology (L3 CAT)
There are total 16 L3 CLOSes. 
The mask should have a continuous value, 
for example: 0x6ff is not valid and 0x600 is valid. 


![](https://software.intel.com/content/dam/develop/external/us/en/images/-p-609896.png)

## MEM Bandwidth Allocation(MBA)
MEM bandwidth can vary from 10% to 100%, 
and you can only increase it by 10%.
There are only 8 CLOSes availiably.
Use `class_id` and `mb_max` to allocate MEM bandwidth to a class.

**The CLOS of MBA corresponds to the CLOS of L3**, 
which means if you map core 0 to CLOS 1, 
core 0 will have L3 CLOS 1 and MBA CLOS 1 together.
And you can only control memory bandwidth of the first 8 L3 CLOSes.

## Monitoring
Current supported monitor event:
```
llc
mbm_local
mbm_total
mbm_remote
mbm_local_delta
mbm_total_delta
mbm_remote_delta
ipc_retired
ipc_retired_delta
ipc_unhalted
ipc_unhalted_deltaypes
ipc
llc_misses
llc_misses_delta
```
The code is in `RDT/monitoring.py`
You can add get these values through `group.values.[name]`.
See `monitoring.py` line 141 - 146 for example.