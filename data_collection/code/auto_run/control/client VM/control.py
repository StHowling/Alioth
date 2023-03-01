import json
import os
import requests
import time
import sys
import subprocess
import re

situation='s5'
flavor='16u'
alphabet_pattern=re.compile(r'([a-zA-Z]+)')
num_pattern=re.compile(r'([0-9|x|a-f]+)')
stress_pattern = re.compile(r'([a-zA-Z\-]+)')
stress_code_map={
    'c':'cpu',
    'C':'cache',
    'S':'socket',
    'd':'hdd',
    'm':'vm',
    'i':'io',
    'L':'LLC_ways',
    'MBW':'memory_bw',
    'CU':'cpu_util'
}

def translate_stressor(code):
    if len(code)==1:
        return stress_code_map[code]
    else:
        return code


def control_task(base_url, cmd_name, app, code):
    if app in ['kafka','etcd'] and code == '1':
        res = requests.delete('%s?command=%s' % (base_url, cmd_name))
    payload = [cmd_name, '/root/%s.py' % cmd_name, app, code]
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

        if app not in ['kafka','etcd'] or code =='1':
            res = requests.delete('%s?command=%s' % (base_url, cmd_name))
        if res.status_code != 200:
            print('failed to stop workload deployment request, '
                  'code: %d' % res.status_code)
        break
    return res.status_code == 200


def clean_result(app_name,stress_code):
    if 'cpu2006' in app_name:
        real_app_name=alphabet_pattern.match(app_name[8:]).group(0)
    else:
        real_app_name=app_name
    src_dir='/root'
    dst_dir=src_dir+'/data-stress-ng/'+real_app_name+'/'+situation+'/'+flavor
    l=0
    if ',' in stress_code:
        stressor_list=stress_code.split(',')
        stressor='/'
        for item in stressor_list[:-1]:
            stressor+=translate_stressor(stress_pattern.match(item).group(0))
            stressor+='+'
        stressor+=(translate_stressor(stress_pattern.match(stressor_list[-1]).group(0))+'/')
    elif stress_code == '0':
        stressor='/'
    else:
        stressor=stress_pattern.match(stress_code).group(0)
        l+=len(stressor)
        stressor='/'+translate_stressor(stressor)+'/'

    dst_dir=dst_dir+stressor

    if ',' in stress_code:
        dst_dir=dst_dir+stress_code+'/'
    else:
        worker_num=num_pattern.match(stress_code[l:]).group(0)
        dst_dir=dst_dir+'w'+worker_num+'/'

    cmd=['mkdir -m 777 -p '+dst_dir]
    p=subprocess.Popen(cmd,shell=True)
    p.wait()

    cmd=['mv '+src_dir+'/2021* '+dst_dir]
    p=subprocess.Popen(cmd,shell=True)
    p.wait()



if __name__ == '__main__':
    if sys.argv[2] == 'stressor':
        lp='170'
        port='8081'
    else:
        lp='233'
        port='8090'
    url = 'http://192.168.0.%s:%s/v1/cmd'%(lp,port)
    name = sys.argv[1]
    if name == 'control_app':
        if not control_task(url, name, sys.argv[2], sys.argv[3]):
            print('failed to perform task: %s' % name)
            exit(1)
    elif name == 'clean_result':
        if 'cpu2006' in sys.argv[2] or 'ffmpeg' in sys.argv[2] or 'stressor' in sys.argv[2]:
            if not control_task(url, name, sys.argv[2], sys.argv[3]):
                print('failed to perform task: %s' % name)
                exit(1)
        else:
            clean_result(sys.argv[2],sys.argv[3])
    else:
        print('usage: control.py control_app|clean_result appname control_code|stress_code')


