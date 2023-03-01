#!/bin/bash

# $1: num of cores    $2 stress type and workers     $3 duration (sec)
#
INSTANCE=instance
PID=0
if [ $1 = '2' ]; then
    INSTANCE="instance-0000015a"
    PID="38217"
elif [ $1 = '4' ]; then
    INSTANCE="instance-00000156"
    PID="7611"
elif [ $1 = '6' ]; then
    INSTANCE="instance-0000015a,instance-00000156"
    PID="38217,7611"
elif [ $1 = '8' ]; then
    INSTANCE="instance-00000769"
    PID="55394"
elif [ $1 = '10' ]; then
    INSTANCE="instance-0000015a,instance-00000769"
    PID="38217,55394"
elif [ $1 = '12' ]; then
    INSTANCE="instance-00000156,instance-00000769"
    PID="7611,55394"
elif [ $1 = '14' ]; then
    INSTANCE="instance-0000015a,instance-00000156,instance-00000769"
    PID="38217,7611,55394"
elif [ $1 = '16' ]; then
    INSTANCE="instance-000003f2"
    PID="73041"
elif [ $1 = '18' ]; then
    INSTANCE="instance-0000015a,instance-000003f2"
    PID="38217,73041"
elif [ $1 = '20' ]; then
    INSTANCE="instance-00000156,instance-000003f2"
    PID="7611,73041"
elif [ $1 = '22' ]; then
    INSTANCE="instance-0000015a,instance-00000156,instance-000003f2"
    PID="38217,7611,73041"
elif [ $1 = '24' ]; then
    INSTANCE="instance-00000769,instance-000003f2"
    PID="55394,73041"
elif [ $1 = '26' ]; then
    INSTANCE="instance-0000015a,instance-00000769,instance-000003f2"
    PID="38217,55394,73041"
elif [ $1 = '28' ]; then
    INSTANCE="instance-00000156,instance-00000769,instance-000003f2"
    PID="7611,55394,73041"
elif [ $1 = '30' ]; then
    INSTANCE="instance-0000015a,instance-00000156,instance-00000769,instance-000003f2"
    PID="38217,7611,55394,73041"
elif [ $1 = '32' ]; then
    INSTANCE="instance-00000184"
    PID="1526"
elif [ $1 = '48' ]; then
    INSTANCE="instance-000003f2,instance-00000184"
    PID="73041,1526"
else 
    echo run-stress-ng.sh vmflavor\(total vcpus\) 0/c/C/S/m\[worker1.worker2.\]\[,stressor2\{workers\},...\] duration\(s\)
    exit 1
fi

curl -X POST http://71.12.106.126:8090/v1/cmd -d '["macro","/home/fsp/czy/model_collect_agent.py","'$INSTANCE'"]'
curl -X POST http://71.12.106.126:8090/v1/cmd -d '["micro","/home/fsp/czy/ShuhaiCollectScript/x86-host/event_collect_multiplexing.py","-I","1","-p","'$PID'","-w","multiplex","-f","/home/fsp/czy/ShuhaiCollectScript/x86-host/chosen_events"]'
curl -X POST http://71.12.106.126:8080/v1/cmd -d '["pqos","/home/fsp/czy/RDT/pqos_collecting.sh","'"$3"'"]'
curl -X POST http://71.12.106.126:8090/v1/cmd -d '["syslog","/home/fsp/czy/log-stressor-stats.py","'$2'","'"$3"'"]'


ip netns exec vpc-af694097-a56a-4817-84ab-e446ad9a6f33 python interact_client_vm_stress.py "stress-$1" $2 $3


curl -X DELETE http://71.12.106.126:8090/v1/cmd?command=macro
curl -X DELETE http://71.12.106.126:8090/v1/cmd?command=micro
curl -X DELETE http://71.12.106.126:8080/v1/cmd?command=pqos
curl -X DELETE http://71.12.106.126:8090/v1/cmd?command=syslog

