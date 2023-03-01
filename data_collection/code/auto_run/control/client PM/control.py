import json
import os
import requests
import time
import sys


def control_task(base_url, cmd_name, app, code):
    payload = [cmd_name, '/root/control.py', cmd_name, app, code]
    resp = requests.post(base_url, data=json.dumps(payload))
    if resp.status_code != 200:
        return False
    while True:
        time.sleep(10)
        res = requests.get('%s?command=%s' % (base_url, cmd_name))
        if res.status_code != 200:
            print('failed to query workload deployment status, '
                  'code: %d' % res.status_code)
            continue

        tokens = res.text.strip().split(',')
        if tokens[0] == 'false':
            continue

        res = requests.delete('%s?command=%s' % (base_url, cmd_name))
        if res.status_code != 200:
            print('failed to stop workload deployment request, '
                  'code: %d' % res.status_code)
        break
    return resp.status_code == 200


if __name__ == '__main__':
    os.environ['no_proxy'] = '*'
    url = 'http://192.168.0.180:8090/v1/cmd'
    name = sys.argv[1]
    if name == 'control_app' or name == 'clean_result':
        if not control_task(url, name, sys.argv[2], sys.argv[3]):
            print('failed to perform intended task: %s' %name)
            exit(1)
    else:
        print('usage: control.py control_app|clean_result appname control_code|stress_code')


