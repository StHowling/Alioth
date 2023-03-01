import subprocess
import os
import sys
import re

situation='s1'
flavor='16u'
alphabet_pattern=re.compile(r'([a-zA-Z]+)')
num_pattern=re.compile(r'([0-9]+)')
stress_pattern = re.compile(r'([a-zA-Z\-]+)')
stress_code_map={
    'c':'cpu',
    'C':'cache',
    'S':'socket',
    'd':'hdd',
    'm':'vm',
    'i':'io'
}

def translate_stressor(code):
    if len(code)==1:
        return stress_code_map[code]
    else:
        return code


if __name__=='__main__':
    app_name=sys.argv[1]
    stress_code=sys.argv[2]
    if 'cpu2006' in app_name:
        real_app_name=alphabet_pattern.match(app_name[8:]).group(0)
    else:
        real_app_name=app_name
    src_dir='/home/fsp/czy'
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

