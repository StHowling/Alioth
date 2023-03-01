import json
import os
import requests
import time
import sys


FINISHED = 'finished'
FAILED = 'failed'


def start_vm_task(base_url, cmd_name, datetime):
    payload = [cmd_name, '/root/align_system_time.py', datetime]
    resp = requests.post(base_url, data=json.dumps(payload))
    return resp.status_code == 200


def stop_vm_task(base_url, cmd_name):
    resp = requests.delete('%s?command=%s' % (base_url, cmd_name))
    return resp.status_code == 200


def wait_vm_task(base_url, cmd_name):
    while True:
        time.sleep(5)
        resp = requests.get('%s?command=%s' % (base_url, cmd_name))
        status, duration = resp.text.strip().split(',')
        if duration == '-1':
            return FAILED
        if status == 'true':
            return FINISHED


if __name__ == '__main__':
    os.environ['no_proxy'] = '*'
    url = 'http://192.168.0.180:8090/v1/cmd'
    name = 'align_time'
    datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    if not start_vm_task(url, name, datetime):
        print('failed to start task')
        exit(1)
    res = wait_vm_task(url, name)
    print(res)
    if not stop_vm_task(url, name):
        print('failed to stop task')
        exit(1)

