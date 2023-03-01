import os
import re
import time


# regular expression to parse needed stat from output file
ab_pattern = re.compile(
    r'timestamp: (\d+), tps: (\d+), delay: (\d+)')
redis_pattern = re.compile(
    r'(\d+)\s+(\d+\.\d+)\s+(\d+)')
sysbench_pattern = re.compile(
    r'.*timestamp: (\d+),.*tps: (\d+\.\d+),.*time: (\d+\.\d+)ms \(95%\)')
rabbitmq_pattern = re.compile(
    r'.*time: (\d+)\.\d+s, sent: (\d+) msg/s, received: (\d+) msg/s.*')
etcd_pattern = re.compile(
    r'.*timestamp: (\d+), delay: (\d+)\.\d+ms, count: (\d+)')
kafka_pattern = re.compile(
    r'timestamp (\d+),.*[(](\d+.\d+) MB/sec[)], (\d+.\d+) ms.*')
mongodb_pattern = re.compile(
    r'.* (\d+) sec: \d+ operations; (\d+|\d+.\d+|\?) '
    r'current ops/sec;.*[(]us[)]: (\d+.\d+).*')


NO_STRESS_OUTPUT_NUM = 4
WITH_STRESS_OUTPUT_NUM = 5

# app constants
APP_REDIS = 'redis'
APP_STORE = 'ab'
APP_MYSQL = 'sysbench'
APP_RABBITMQ = 'rabbitmq'
APP_ETCD = 'etcd'
APP_KAFKA = 'kafka'
APP_MONGODB = 'mongoDB'
APP_FFMPEG = 'ffmpeg'

# stress type constants
STRESS_NONE = '0'
STRESS_CPU = 'c'
STRESS_CACHE = 'C'
STRESS_MEM = 'm'
STRESS_SOCKET = 'S'


def _parse_app_by_regex(path, contain_key, pattern, metric_str):
    time_metrics_map = {}
    with open(path) as f:
        for line in f:
            if contain_key and contain_key not in line:
                continue
            match = re.match(pattern, line.strip())
            if match:
                time_metrics_map[match.group(1)] = '%s,%s' % (
                    match.group(2), match.group(3))
    return time_metrics_map, metric_str


def _add_start_time_to_key(time_metrics_map, path):
    file_name = os.path.basename(path)
    time_str = file_name.split('-')[0]
    time_tick = time.mktime(time.strptime(time_str, "%Y%m%d%H%M%S"))
    return {'%d' % (time_tick+int(t)): metrics
            for t, metrics in time_metrics_map.items()}


def parse_ab(path):
    return _parse_app_by_regex(path, 'timestamp', ab_pattern, 'tps,delay')


def parse_redis(path):
    return _parse_app_by_regex(path, '', redis_pattern, 'latency,count')


def parse_sysbench(path):
    return _parse_app_by_regex(path, 'timestamp', sysbench_pattern,
                               'tps,response_time')


def parse_rabbitmq(path):
    time_metrics_map, metrics_str = _parse_app_by_regex(
        path, 'time', rabbitmq_pattern, 'sent_speed,received_speed')
    return _add_start_time_to_key(time_metrics_map, path), metrics_str


def parse_etcd(path):
    return _parse_app_by_regex(path, 'timestamp', etcd_pattern,
                               'latency,count')


def parse_kafka(path):
    return _parse_app_by_regex(path, 'timestamp', kafka_pattern, 'tps,latency')


def parse_mongodb(path):
    time_metrics_map, metrics_str = _parse_app_by_regex(
        path, 'est completion', mongodb_pattern, 'count,latency')
    return _add_start_time_to_key(time_metrics_map, path), metrics_str


def parse_ffmpeg(path):
    time_metrics_map = {}
    last_timestamp = None
    with open(path) as f:
        for line in f:
            timestamp, speed = line.strip().split(',')
            if not last_timestamp:
                last_timestamp = float(timestamp)
                continue
            time_metrics_map[timestamp.split('.')[0]] = '%.4f' % (
                int(speed) / (float(timestamp) - last_timestamp))
            last_timestamp = float(timestamp)
    return time_metrics_map, 'speed'


def parse_ffmpeg_pv(path):
    time_metrics_map = {}
    file_name = os.path.basename(path)
    time_str = file_name.split('-')[0]
    time_tick = time.mktime(time.strptime(time_str, "%Y%m%d%H%M%S"))
    with open(path) as f:
        for c, line in enumerate(f):
            start = line.find('[')
            end = line.find('MiB')
            time_metrics_map['%d' % (time_tick+c+1)] = line[start+1:end]
    return time_metrics_map, 'speed'


def parse_macro_metrics(path):
    return _parse_metrics(path, 'timestamp', ',', 2)


def parse_micro_metrics(path):
    return _parse_metrics(path, 'Time', ';', 1)


def _parse_metrics(path, time_key, split_char, start):
    time_metrics_map = {}
    metrics = ''
    with open(path) as f:
        for line in f:
            if time_key in line:
                metrics = ','.join(line.strip().split(split_char)[start:])
                continue
            tokens = line.strip().split(split_char)
            time_metrics_map[tokens[0]] = ','.join(tokens[start:])
    return time_metrics_map, metrics


def parse_stress(path):
    range_list = []
    with open(path) as f:
        start = -1
        for line in f:
            if line.startswith('start'):
                start = int(line.strip().split(' ')[2])
            elif line.startswith('end'):
                if start == -1:
                    print('unmatched end')
                    return range_list
                end = int(line.strip().split(' ')[2])
                range_list.append((start, end))
                start = -1
    return range_list


def merge_map(map1, map2):
    merged_map = {}
    for k, v in map1.items():
        if k not in map2:
            continue
        merged_map[k] = '%s,%s' % (v, map2[k])
    return merged_map


def hey_aggregate():
    counts = [-1] * 100
    with open('D:/MLdata/log-csv.csv') as f:
        for line in f:
            if 'DNS' in line:
                continue
            tokens = line.strip().split(',')
            counts[int(float(tokens[7]))] += 1
    for c in counts:
        print(c)


def full_extract_stress_range(base_path, file_names, apps):
    time_path_pairs = []
    for file_name in file_names:
        if 'stress' not in file_name:
            continue

        time_str = file_name.split('-')[0]
        time_tick = time.mktime(time.strptime(time_str, "%Y%m%d%H%M%S"))
        time_path_pairs.append((time_tick, os.path.join(base_path, file_name)))
    time_path_pairs = sorted(time_path_pairs, key=lambda x: x[0])
    app_range_map = {}
    for idx, app in enumerate(apps):
        app_range_map[app] = parse_stress(time_path_pairs[idx][1])
    return app_range_map


def merge_stress_label(time_metrics_map, stress_ranges):
    full_start = int(min(time_metrics_map.keys()))
    full_end = int(max(time_metrics_map.keys()))
    time_stress_map = dict(('%d' % timestamp, 0) for timestamp in range(
        full_start, full_end + 1))
    for (start, end) in stress_ranges:
        for timestamp in range(start, end):
            time_stress_map['%d' % timestamp] = 1
    for timestamp in time_metrics_map:
        metric_str = time_metrics_map[timestamp]
        time_metrics_map[timestamp] = "%s,%d" % (metric_str,
                                                 time_stress_map[timestamp])


# app and corresponding output parsing function map
# the functions signature is func(path: str) -> (Dict[str, str], str)

# the key of the returned dict is unix timestamp, the value of the returned
# dict is the performance at the corresponding timestamp, if multiple
# performance metrics are collected, join the metric values with ","
# take ab as example, the value will be "value_of_tps,value_of_delay"

# the second return value is the name of performance metrics, also joined with
# ",", still take ab as example, the value will be "tps,delay"
app_func_map = {
    APP_REDIS: parse_redis,
    APP_STORE: parse_ab,
    APP_MYSQL: parse_sysbench,
    APP_RABBITMQ: parse_rabbitmq,
    APP_ETCD: parse_etcd,
    APP_KAFKA: parse_kafka,
    APP_MONGODB: parse_mongodb,
    APP_FFMPEG: parse_ffmpeg
}


def full_merge(base_path, file_names, apps, stress_label):
    file_map = {}
    for app in app_func_map:
        for key in ('load', 'macro', 'micro'):
            file_map['%s_%s' % (app, key)] = ''
    time_path_pairs = []
    for file_name in file_names:
        path = os.path.join(base_path, file_name)
        tokens = file_name.split('-')
        for app in app_func_map:
            if app in tokens:
                file_map['%s_load' % app] = path
                break
        else:
            if 'events.log' not in file_name and 'stress' not in file_name:
                time_str = file_name.split('-')[0]
                time_tick = time.mktime(time.strptime(time_str,
                                                      "%Y%m%d%H%M%S"))
                time_path_pairs.append((time_tick, path))
    time_path_pairs = sorted(time_path_pairs, key=lambda x: x[0])
    cursor = 0
    while cursor + 1 < len(time_path_pairs):
        if 'events' in time_path_pairs[cursor][1]:
            file_map['%s_micro' % apps[cursor // 2]] = \
                time_path_pairs[cursor][1]
            file_map['%s_macro' % apps[cursor // 2]] = \
                time_path_pairs[cursor + 1][1]
        else:
            file_map['%s_micro' % apps[cursor // 2]] = \
                time_path_pairs[cursor + 1][1]
            file_map['%s_macro' % apps[cursor // 2]] = \
                time_path_pairs[cursor][1]
        cursor += 2

    if stress_label:
        app_stress_range_map = full_extract_stress_range(
            base_path, file_names, apps)

    for app in apps:
        time_load_metrics_map, load_metrics = app_func_map[app](
            file_map['%s_load' % app])
        time_macro_metrics_map, macro_metrics = parse_macro_metrics(
            file_map['%s_macro' % app])
        time_micro_metrics_map, micro_metrics = parse_micro_metrics(
            file_map['%s_micro' % app])
        final_map = merge_map(time_macro_metrics_map, time_micro_metrics_map)
        final_map = merge_map(final_map, time_load_metrics_map)

        if stress_label:
            merge_stress_label(final_map, app_stress_range_map[app])

        with open(base_path + '/%s-merged.csv' % app, 'w') as wf:
            head = 'timestamp,%s,%s,%s' % (
                macro_metrics, micro_metrics, load_metrics)

            if stress_label:
                head += ',with_stress'

            head += '\n'

            wf.write(head)
            for timestamp, all_metrics in final_map.items():
                wf.write('%s,%s\n' % (timestamp, all_metrics))


def group_data_files(base_path, stress_switches, app_num):
    """
    :param base_path: base path for data files
    :param stress_switches: list of stress type indicating what stress is
    added in each run
    :param app_num: number of apps in each run
    :type stress_switches: list[str]
    """
    time_name_pairs = []
    file_groups = []
    for file_name in os.listdir(base_path):
        time_str = file_name.split('-')[0]
        time_tick = time.mktime(time.strptime(time_str, "%Y%m%d%H%M%S"))
        time_name_pairs.append((time_tick, file_name))
    sorted_pairs = sorted(time_name_pairs, key=lambda x: x[0])
    cursor = 0
    for switch in stress_switches:
        if switch != '0':
            c = WITH_STRESS_OUTPUT_NUM * app_num
        else:
            c = NO_STRESS_OUTPUT_NUM * app_num
        file_groups.append([name for _, name in sorted_pairs[cursor:cursor+c]])
        cursor += c
    return file_groups


stress_code_label_map = {
    STRESS_NONE: 'no_stress',
    STRESS_CPU: 'cpu_stress',
    STRESS_CACHE: 'cache_stress',
    STRESS_MEM: 'mem_stress',
    STRESS_SOCKET: 'socket_stress'
}


if __name__ == '__main__':
    # work_dir and output_dir should be different
    work_dir = 'D:/ML-data/4-colocation-0409/run-2/'+APP_RABBITMQ
    output_dir = 'D:/ML-data/4-colocation-0409/run-2/merged'
    repeat_num = 1
    switches = [STRESS_NONE] * repeat_num
    targets = [APP_RABBITMQ]
    groups = group_data_files(work_dir, switches, len(targets))
    for i, group in enumerate(groups):
        full_merge(work_dir, group, targets, switches[i] != '0')
        label = stress_code_label_map[switches[i]]
        for target in targets:
            os.rename('%s/%s-merged.csv' % (work_dir, target),
                      '%s/%s-merged-%s-%d.csv' % (output_dir, target,
                                                  label, i + 1))
