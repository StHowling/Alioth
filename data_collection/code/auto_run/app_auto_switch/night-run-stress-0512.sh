#!/bin/bash

############## To make full use of overnight time, we may need to switch
############## app to test once the former finishes.
############## This is an example of produing a long run with app switching.
############## Note that control_app.py is an initial version, most apps
############## are not included. Yet you can implement methods in Frequent_shell_scripts.md
############## yourself and push to this repository.
############## Later the control.py can incorporate load_vm_flavor.py
############## to automatically change core binding settings if needed.

#bash batch-run-2021051110.sh


ip netns exec vpc-af694097-a56a-4817-84ab-e446ad9a6f33 python control.py control_app rabbitmq 0

bash stress_ng_repeat_3.sh rabbitmq C4
bash stress_ng_repeat_3.sh rabbitmq C8
bash stress_ng_repeat_3.sh rabbitmq C16
bash stress_ng_repeat_3.sh rabbitmq C32


bash stress_ng_repeat_3.sh rabbitmq c4
bash stress_ng_repeat_3.sh rabbitmq c8
bash stress_ng_repeat_3.sh rabbitmq c16
bash stress_ng_repeat_3.sh rabbitmq c32


bash stress_ng_repeat_3.sh rabbitmq m4
bash stress_ng_repeat_3.sh rabbitmq m8
bash stress_ng_repeat_3.sh rabbitmq m16
bash stress_ng_repeat_3.sh rabbitmq m32


bash stress_ng_repeat_3.sh rabbitmq d4
bash stress_ng_repeat_3.sh rabbitmq d8
bash stress_ng_repeat_3.sh rabbitmq d16
bash stress_ng_repeat_3.sh rabbitmq d32


bash stress_ng_repeat_3.sh rabbitmq S4
bash stress_ng_repeat_3.sh rabbitmq S8
bash stress_ng_repeat_3.sh rabbitmq S16
bash stress_ng_repeat_3.sh rabbitmq S32


ip netns exec vpc-af694097-a56a-4817-84ab-e446ad9a6f33 python control.py control_app rabbitmq 1
sleep 10
ip netns exec vpc-af694097-a56a-4817-84ab-e446ad9a6f33 python control.py control_app etcd 0

bash stress_ng_repeat_3.sh etcd C4
bash stress_ng_repeat_3.sh etcd C8
bash stress_ng_repeat_3.sh etcd C16
bash stress_ng_repeat_3.sh etcd C32


bash stress_ng_repeat_3.sh etcd c4
bash stress_ng_repeat_3.sh etcd c8
bash stress_ng_repeat_3.sh etcd c16
bash stress_ng_repeat_3.sh etcd c32


bash stress_ng_repeat_3.sh etcd m4
bash stress_ng_repeat_3.sh etcd m8
bash stress_ng_repeat_3.sh etcd m16
bash stress_ng_repeat_3.sh etcd m32


bash stress_ng_repeat_3.sh etcd d4
bash stress_ng_repeat_3.sh etcd d8
bash stress_ng_repeat_3.sh etcd d16
bash stress_ng_repeat_3.sh etcd d32

bash stress_ng_repeat_3.sh etcd S4
bash stress_ng_repeat_3.sh etcd S8
bash stress_ng_repeat_3.sh etcd S16
bash stress_ng_repeat_3.sh etcd S32

ip netns exec vpc-af694097-a56a-4817-84ab-e446ad9a6f33 python control.py control_app etcd 1
