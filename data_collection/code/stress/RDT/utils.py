import os

def bytes_to_kb(num_bytes):
    return num_bytes / 1024.0


def bytes_to_mb(num_bytes):
    return num_bytes / (1024.0 * 1024.0)

def add_process_to_cgroup():
    pid = os.getpid()
    os.system("echo %d >> /sys/fs/cgroup/cpuset/test/tasks" %(pid))
