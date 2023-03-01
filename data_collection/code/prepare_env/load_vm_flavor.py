import sys
import json
import os

domain2pth = {
    # determine this part on the specific PM
    # '3':'x2d3\\\\x2dinstance\\\\x2d00000156.scope/',
    # '4':'x2d4\\\\x2dinstance\\\\x2d00000159.scope/',
    # '5':'x2d5\\\\x2dinstance\\\\x2d0000015a.scope/',
    # '7':'x2d7\\\\x2dinstance\\\\x2d00000184.scope/',
    # '8':'x2d8\\\\x2dinstance\\\\x2d000003f2.scope/',
    # '9':'x2d9\\\\x2dinstance\\\\x2d0000074a.scope/',
    # '10':'x2d10\\\\x2dinstance\\\\x2d0000074b.scope/',
    # '11':'x2d11\\\\x2dinstance\\\\x2d0000074c.scope/',
    # '12':'x2d12\\\\x2dinstance\\\\x2d0000074d.scope/',
    # '13':'x2d13\\\\x2dinstance\\\\x2d0000074e.scope/',
    # '14':'x2d14\\\\x2dinstance\\\\x2d00000769.scope/'
}
domain2pth_raw = {
    # same content, but without escape characters, for use in os.listdir()
}
pth = '/sys/fs/cgroup/cpuset/machine.slice/machine-qemu\\\\%s%s/cpuset.cpus'


def load_flavor(config_file):
    with open(config_file) as f:
        content = f.read()
        config = json.loads(content)

    for did, flavor in config.items():
        #with open(pth%(domain2pth[did], 'emulator'), 'w') as fout:
        #    fout.write(flavor['emulator'])
        os.system("echo \"%s\" > %s"%(flavor['emulator'], pth%(domain2pth[did], 'emulator')))
        print(did, 'emulator->', flavor['emulator'])
        for item in os.listdir('/sys/fs/cgroup/cpuset/machine.slice/machine-qemu\\%s'%domain2pth_raw[did]):
            #with open(pth%(domain2pth[did], 'iothread'+str(i)), 'w') as fout:
            #    fout.write(flavor['iothread'])
            if 'iothread' in item:
                os.system("echo \"%s\" > %s"%(flavor['iothread'], pth%(domain2pth[did], item)))
                print(did, item +'->', flavor['iothread'])

        for vids, cores in flavor['vcpu'].items():
            for vid in range(int(vids.split('-')[0]),int(vids.split('-')[1])+1):
                #with open(pth%(domain2pth[did], 'vcpu'+str(vid)), 'w') as fout:
                #    fout.write(cores)
                os.system("echo \"%s\" > %s"%(cores, pth%(domain2pth[did], 'vcpu'+str(vid))))
                print(did, 'vcpu'+str(vid)+'->', cores)

if __name__ == '__main__':
    load_flavor(sys.argv[1])