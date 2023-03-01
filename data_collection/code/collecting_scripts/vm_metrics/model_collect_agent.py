import json
import logging
import re
import subprocess
import signal
import sys
import time

import libvirt
from xml.etree import ElementTree as et

CPU_TOTAL = 'cpu_time'
CPU_SYSTEM = 'system_time'
CPU_USER = 'user_time'
MEM_AVAILABLE = 'available'
MEM_ACTUAL = 'actual'
MEM_UNUSED = 'unused'
NET_RD_BYTE = 'net_rd_byte'
NET_WR_BYTE = 'net_wr_byte'
NET_RD_PACKET = 'net_rd_packet'
NET_WR_PACKET = 'net_wr_packet'
BLK_RD_BYTE = 'rd_bytes'
BLK_WR_BYTE = 'wr_bytes'
BLK_RD_OP = 'rd_operations'
BLK_WR_OP = 'wr_operations'
BLK_RD_TIME = 'rd_total_times'
BLK_WR_TIME = 'wr_total_times'
BLK_FLUSH_TIME = 'flush_total_times'
CPU_METRICS = (CPU_TOTAL, CPU_SYSTEM, CPU_USER)
MEM_METRICS = (MEM_AVAILABLE, MEM_ACTUAL, MEM_UNUSED)
NET_METRICS = (NET_RD_BYTE, NET_WR_BYTE, NET_RD_PACKET, NET_WR_PACKET)
BLK_METRICS = (BLK_RD_BYTE, BLK_WR_BYTE, BLK_RD_OP, BLK_WR_OP, BLK_RD_TIME,
               BLK_WR_TIME, BLK_FLUSH_TIME)
METRICS = (
    CPU_TOTAL, CPU_SYSTEM, CPU_USER, MEM_AVAILABLE, MEM_ACTUAL, MEM_UNUSED,
    BLK_RD_BYTE, BLK_WR_BYTE, BLK_RD_OP, BLK_WR_OP, BLK_RD_TIME, BLK_WR_TIME,
    BLK_FLUSH_TIME, NET_RD_BYTE, NET_WR_BYTE, NET_RD_PACKET, NET_WR_PACKET
)
APP_NAME = 'collector'
RETRIES = 3


def setup_logger(name, path, level):
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path)
    fh.setLevel(getattr(logging, level.upper()))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)


class Collector(object):
    def __init__(self):
        self.vm_pid_map = {}
        self.vm_nics_map = {}
        self.nic_vm_map = {}
        self.cache = {}

        self.rpacket_pattern = re.compile('RX packets:(\\d+)')
        self.wpacket_pattern = re.compile('TX packets:(\\d+)')
        self.byte_pattern = re.compile('RX bytes:(\\d+).*TX bytes:(\\d+)')

        self.conn = self._get_libvirt_conn()
        self.logger = logging.getLogger(APP_NAME)

    def _get_libvirt_conn(self):
        for i in range(RETRIES):
            try:
                conn = libvirt.open('qemu:///system')
                return conn
            except:
                if i == RETRIES - 1:
                    raise
                self.logger.error('failed to open libvirt connection, retry')
                time.sleep(0.1)

    def _get_command_output(self, args, retries):
        if len(args) == 0:
            return '', False
        for i in range(retries):
            try:
                return subprocess.check_output(args), True
            except subprocess.CalledProcessError:
                self.logger.error('failed to retrieve net stat by %s, '
                                  'try %d time(s)' % (args[0], i + 1))
        return '', False

    def _calc_diff(self, current_timestamp, key, value):
        if key in self.cache:
            last_timestamp, last_value = self.cache[key]
            diff = (value - last_value) / (current_timestamp - last_timestamp)
        else:
            diff = 0
        self.cache[key] = (current_timestamp, value)
        return diff

    def _get_dom_nics(self, dom):
        nics = []
        try:
            domain_config = et.fromstring(dom.XMLDesc())
            ifaces = domain_config.findall('devices/interface/target')
            for i in ifaces:
                nic = i.get('dev')
                nics.append(nic)
            return nics
        except:
            self.logger.error(
                'get nics from vm %s exception' % dom.UUIDString())
            return nics

    def _collect_net_ovs(self, current_timestamp, nic_vm_map, vm_net_map):
        _vm_net_map = {}
        output, valid = self._get_command_output(
            ['ovs-appctl', 'dpctl/show', '--statistics'], RETRIES)
        if not valid:
            return
        lines = output.split('\n')
        line_index = 0
        block_lines = 5

        while line_index < len(lines):
            line = lines[line_index]
            if 'port' in line:
                nic_name = line.lstrip().split(' ')[2]
                if nic_name not in nic_vm_map:
                    line_index += block_lines
                    continue
                r_packet_m = self.rpacket_pattern.search(lines[line_index + 1])
                w_packet_m = self.wpacket_pattern.search(lines[line_index + 2])
                byte_m = self.byte_pattern.search(lines[line_index + 4])
                if not all([r_packet_m, w_packet_m, byte_m]):
                    self.logger.error(
                        'packet/byte not found in ovs-appctl output')
                    line_index += block_lines
                    continue
                r_packet = float(r_packet_m.group(1))
                w_packet = float(w_packet_m.group(1))
                r_byte = float(byte_m.group(1))
                w_byte = float(byte_m.group(2))

                libvirt_name = nic_vm_map.get(nic_name)
                if libvirt_name not in _vm_net_map:
                    _vm_net_map[libvirt_name] = {}
                    for metric in NET_METRICS:
                        _vm_net_map[libvirt_name][metric] = 0
                _vm_net_map[libvirt_name][NET_RD_BYTE] += r_byte
                _vm_net_map[libvirt_name][NET_WR_BYTE] += w_byte
                _vm_net_map[libvirt_name][NET_RD_PACKET] += r_packet
                _vm_net_map[libvirt_name][NET_WR_PACKET] += w_packet

                line_index += block_lines
            else:
                line_index += 1

        for libvirt_name in vm_net_map:
            if libvirt_name not in _vm_net_map:
                for metric in NET_METRICS:
                    vm_net_map[libvirt_name][metric] = 0
            else:
                for metric in NET_METRICS:
                    vm_net_map[libvirt_name][metric] = self._calc_diff(
                        current_timestamp, libvirt_name + metric,
                        _vm_net_map[libvirt_name][metric])

    def collect(self, libvirt_names):
        vm_metrics_map = {k: dict() for k in libvirt_names}
        current_timestamp = time.time()

        for libvirt_name in libvirt_names:
            dom = self.conn.lookupByName(libvirt_name)
            if not dom or not dom.isActive():
                continue

            # cpu stat
            cpu_stat = dom.getCPUStats(True)[0]
            for metric in CPU_METRICS:
                vm_metrics_map[libvirt_name][metric] = self._calc_diff(
                    current_timestamp * 1000000000, libvirt_name + metric,
                    cpu_stat[metric]) * 100

            # mem stat
            mem_stat = dom.memoryStats()
            for metric in MEM_METRICS:
                vm_metrics_map[libvirt_name][metric] = mem_stat[metric]

            # net stat
            if libvirt_name not in self.vm_nics_map:
                nics = self._get_dom_nics(dom)
                self.vm_nics_map[libvirt_name] = nics
                for nic in nics:
                    self.nic_vm_map[nic] = libvirt_name
            self._collect_net_ovs(current_timestamp, self.nic_vm_map,
                                  vm_metrics_map)

            # blk stat
            blk_stats = dom.blockStatsFlags('', 0)
            for metric in BLK_METRICS:
                vm_metrics_map[libvirt_name][metric] = self._calc_diff(
                    current_timestamp, libvirt_name + metric,
                    blk_stats[metric])

        return current_timestamp, vm_metrics_map


class Runner(object):
    def __init__(self):
        self.stopped = False
        signal.signal(signal.SIGTERM, self.signal_handle)

    def signal_handle(self, signum, frame):
        self.stopped = True

    def collect_loop(self, libvirt_names, interval, loop_num, pesist_path):
        collector = Collector()
        loop_count = 0
        with open(pesist_path, 'w') as wf:
            wf.write('timestamp,name,%s\n' % ','.join(METRICS))
            while True:
                loop_count += 1
                if loop_count > loop_num > 0:
                    break
                if self.stopped:
                    break
                try:
                    (current_timestamp,
                     vm_metrics_map) = collector.collect(libvirt_names)
                    for vm, metrics_map in vm_metrics_map.items():
                        values = [metrics_map[metric] for metric in METRICS]
                        line = ','.join(('%.2f' % value for value in values))
                        wf.write('%d,%s,%s\n' % (current_timestamp, vm, line))
                except KeyboardInterrupt:
                    collector.logger.info(
                        'collect loop stopped by keyboard interrupt')
                    break
                except Exception as e:
                    collector.logger.warning(
                        'collect loop failed becase %s, will retry' % e)
                    # any problem occurs, we recreate collector and try again
                    collector = Collector()
                time.sleep(interval)


if __name__ == '__main__':
    with open('config.json') as f:
        content = f.read()
        config = json.loads(content)
    setup_logger(APP_NAME, config['log_path'], config['log_level'])
    runner = Runner()
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    runner.collect_loop([sys.argv[1]], config['interval'],
                        config['count'], '%s-metrics.csv' % timestamp)
