import time

from pqos import Pqos
from pqos.cpuinfo import PqosCpuInfo
from pqos.capability import PqosCap, CPqosMonitor
from pqos.monitoring import PqosMon
from utils import bytes_to_kb, bytes_to_mb, add_process_to_cgroup

def get_event_name(event_type):
    """
    Converts a monitoring event type to a string label required by libpqos
    Python wrapper.

    Parameters:
        event_type: monitoring event type

    Returns:
        a string label
    """

    event_map = {
        CPqosMonitor.PQOS_MON_EVENT_L3_OCCUP: 'l3_occup',
        CPqosMonitor.PQOS_MON_EVENT_LMEM_BW: 'lmem_bw',
        CPqosMonitor.PQOS_MON_EVENT_TMEM_BW: 'tmem_bw',
        CPqosMonitor.PQOS_MON_EVENT_RMEM_BW: 'rmem_bw',
        CPqosMonitor.PQOS_PERF_EVENT_LLC_MISS: 'perf_llc_miss',
        CPqosMonitor.PQOS_PERF_EVENT_IPC: 'perf_ipc'
    }

    return event_map.get(event_type)

def get_supported_events():
    """
    Returns a list of supported monitoring events.

    Returns:
        a list of supported monitoring events
    """

    cap = PqosCap()
    mon_cap = cap.get_type('mon')

    events = [get_event_name(event.type) for event in mon_cap.events]

    return events


def get_all_cores():
    """
    Returns all available cores.

    Returns:
        a list of available cores
    """

    cores = []
    cpu = PqosCpuInfo()
    sockets = cpu.get_sockets()

    for socket in sockets:
        cores += cpu.get_cores(socket)

    return cores


class Monitoring:
    "Generic class for monitoring"

    def __init__(self):
        self.mon = PqosMon()
        self.groups = []

    def setup_groups(self):
        "Sets up monitoring groups. Needs to be implemented by a derived class."

        return []

    def setup(self):
        "Resets monitoring and configures (starts) monitoring groups."

        self.mon.reset()
        self.groups = self.setup_groups()

    def update(self):
        "Updates values for monitored events."

        self.mon.poll(self.groups)

    def print_data(self):
        """Prints current values for monitored events. Needs to be implemented
        by a derived class."""

        pass

    def stop(self):
        "Stops monitoring."

        for group in self.groups:
            group.stop()


class MonitoringCore(Monitoring):
    "Monitoring per core"

    def __init__(self, cores, events):
        """
        Initializes object of this class with cores and events to monitor.

        Parameters:
            cores: a list of cores to monitor
            events: a list of monitoring events
        """

        super(MonitoringCore, self).__init__()
        self.cores = cores or get_all_cores()
        self.events = events

    def setup_groups(self):
        """
        Starts monitoring for each core using separate monitoring groups for
        each core.

        Returns:
            created monitoring groups
        """

        groups = []

        for core in self.cores:
            group = self.mon.start([core], self.events)
            groups.append(group)

        return groups

    def print_data(self):
        "Prints current values for monitored events."

        print("    CORE    LLC[KB]    MBL[MB]    MBR[MB]")

        for group in self.groups:
            core = group.cores[0]
            #rmid = group.poll_ctx[0].rmid if group.poll_ctx else 'N/A'
            llc = bytes_to_kb(group.values.llc)
            mbl = bytes_to_mb(group.values.mbm_local_delta)
            mbr = bytes_to_mb(group.values.mbm_remote_delta)
            print("%8u %10.1f %10.1f %10.1f" % (core, llc, mbl, mbr))


class PqosContextManager:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pqos = Pqos()

    def __enter__(self):
        self.pqos.init(*self.args, **self.kwargs)
        return self.pqos
    
    def __exit__(self, *args, **kwargs):
        self.pqos.fini()
        return None


def main():
    add_process_to_cgroup()
    COREID = [0,2,5]
    with PqosContextManager("MSR"):
        events = get_supported_events()
        monitoring = MonitoringCore(COREID, events)
        monitoring.setup()

        while True:
            try:
                monitoring.update()
                monitoring.print_data()

                time.sleep(1.0)
            except KeyboardInterrupt:
                break

        monitoring.stop()

if __name__ == "__main__":
    main()