#!/usr/bin/env python
# -*- coding:utf-8 -*-
# __author__ = 'wlm'
# Copyright 2017 HuaWei Tld
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

# please refer to http://3ms.huawei.com/km/groups/3945264/blogs/details/9296541
# for more information of this tool

import csv
from datetime import datetime, timedelta
import getopt
import logging
import signal
import subprocess
import sys
import time


def print_help():
    print('python event_collect.py\n'
          '-C <core>              default = None\n'
          '-p <pid>               default is to collect for entire system\n'
          '-f <event-file>        default = total_events\n'
          '-I <interval-seconds>  default = 1\n'
          '-m <mode>              default = host, support: host and kvm\n'
          '-w <workload_name>     default = test \n')


if __name__ == '__main__':
    # get script input
    cores = None
    input_pid = None
    intervalPrint = 1000
    workload_name = "test"
    input_mode = "host"
    event_file = "total_events"

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hp:C:I:m:w:f:",
                                   ["help", "output="])
    except getopt.GetoptError:
        print('python event_collect.py -C <core> '
              '-p <pid> -m <mode> -f <event-file> '
              '-I <interval-seconds> -w <workload-name> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-C", "--core"):
            cores = str(arg)
        elif opt in ("-p", "--pid"):
            input_pid = int(arg)
        elif opt in ("-f", "--event-file"):
            event_file = arg
        elif opt in ("-I", "--interval-seconds"):
            intervalPrint = int(float(arg)*1000)
        elif opt in ("-m", "--mode"):
            input_mode = arg
            if input_mode not in ["host", "kvm"]:
                print("please input invalid mode: host or kvm !\n")
                sys.exit()
        elif opt in ("-w", "--workload-name"):
            workload_name = arg

    # generate perf event table, judge loop circles
    EVENT_FILENAME = event_file
    total_events_list = list()
    with open(EVENT_FILENAME, "r") as eventsFile:
        for line in eventsFile.readlines():
            event = line.strip()
            total_events_list.append(event)

    if not total_events_list:
        raise Exception("event file is empty")

    last_event = total_events_list[-1]

    # prepare csv header
    timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    LOG_FILENAME = timeStamp + "-" + workload_name + "-events.log"
    CSV_FILENAME = timeStamp + "-" + workload_name + "-events.csv"
    logging.basicConfig(
        filename=LOG_FILENAME, level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d  %H:%M:%S %a')

    csv_headers = total_events_list[:]
    csv_headers.insert(0, "cycles")
    csv_headers.insert(0, "instructions")
    csv_headers.insert(0, "Time")
    with open(CSV_FILENAME, 'w') as csvFile:
        writer = csv.DictWriter(csvFile, csv_headers, delimiter=";")
        writer.writeheader()

    # call perf command
    perf_mode = "perf stat"
    if input_mode == "host":
        perf_mode = "perf stat"
    elif input_mode == "kvm":
        perf_mode = "perf kvm stat"

    perf_cmd = perf_mode + " -I " + str(intervalPrint) + " -e " + \
                           " -e ".join(total_events_list)

    if input_pid is not None:
        perf_cmd = perf_cmd + " -p " + str(input_pid)
    if cores is not None:
        perf_cmd = perf_cmd + " -C " + str(cores)

    perf_cmd = perf_cmd + " -a "
    logging.info(perf_cmd)

    pPerf = None
    rc = None
    for retry_time in range(3):
        pPerf = subprocess.Popen([perf_cmd], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, shell=True)
        start_time = datetime.now()
        time.sleep(1)
        rc = subprocess.Popen.poll(pPerf)
        if not rc:
            break

    if rc:
        stdout, stderr = pPerf.communicate()
        logging.error(stderr)
        raise Exception(stderr)

    # handle signal
    def main_handler(signal, frame):
        pPerf.kill()
        pPerf.wait()
        sys.exit(0)

    def child_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, main_handler)
    signal.signal(signal.SIGTERM, main_handler)
    signal.signal(signal.SIGCHLD, child_handler)

    # read perf output
    csv_event_value = dict()
    with open(CSV_FILENAME, 'a') as csvFile:
        writer = csv.DictWriter(csvFile, csv_headers, delimiter=";")
        try:
            while True:
                line = pPerf.stderr.readline()

                # skip lines with label or error
                if line.startswith("failed"):
                    continue
                elif line.split()[1].startswith('#'):
                    continue
                elif line.startswith('#'):
                    continue

                # get metric value
                event_name = line.split()[2]
                if event_name == 'counted>':
                    event_name = line.split()[3]
                event_value = line.split()[1].replace(',', '')
                csv_event_value[event_name] = event_value

                if event_name == last_event:
                    # get collect timestamp
                    elapsed_time_raw = line.split()[0].strip()
                    elapsed_time = int(float(elapsed_time_raw))
                    now_time = start_time + timedelta(seconds=elapsed_time)
                    csv_event_value['Time'] = int(time.mktime(
                        now_time.timetuple()))

                    # output to file
                    try:
                        writer.writerow(csv_event_value)
                    except ValueError as ex:
                        logging.error(ex)

                    csv_event_value = dict()

        finally:
            pPerf.kill()
            pPerf.wait()
