import subprocess
import sys
import datetime
from multiprocessing import Pool

args = ['/bin/bash','./run-cpu2006.sh']

def run_app(app_name, idx):
    t = datetime.datetime.now()
    x = t.strftime('%Y%m%d%H%M%S')
    args.append(app_name)
    with open('%s-%s-result%s.log' % (x,app_name,idx),'w') as f:
        p = subprocess.Popen(args, stdout=f)
        p.wait()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        worker_num=4
    else: 
        worker_num=int(sys.argv[2])
    pool = Pool(processes=worker_num)
    for i in range(worker_num):
        pool.apply_async(run_app, args=(sys.argv[1], i))
    pool.close()
    pool.join()
