from pqos import Pqos
from pqos.cpuinfo import PqosCpuInfo
from pqos.mba import PqosMba

def set_allocation_class(sockets, class_id, mb_max):
    """
    Sets up allocation classes of service on selected CPU sockets

    Parameters:
        sockets: array with socket IDs
        class_id: class of service ID
        mb_max: COS rate in percent
    """

    mba = PqosMba()
    cos = mba.COS(class_id, mb_max)

    for socket in sockets:
        try:
            actual = mba.set(socket, [cos])

            params = (socket, class_id, mb_max, actual[0].mb_max)
            print("SKT%u: MBA COS%u => %u%% requested, %u%% applied" % params)
        except:
            print("Setting up cache allocation class of service failed!")


def print_allocation_config(sockets):
    """
    Prints allocation configuration.

    Parameters:
        sockets: array with socket IDs
    """

    mba = PqosMba()

    for socket in sockets:
        try:
            coses = mba.get(socket)

            print("MBA COS definitions for Socket %u:" % socket)

            for cos in coses:
                cos_params = (cos.class_id, cos.mb_max)
                print("    MBA COS%u => %u%% available" % cos_params)
        except:
            print("Error")
            raise


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
    class_id = 0
    mb_max = 100

    try:
        with PqosContextManager("MSR"):
            cpu = PqosCpuInfo()
            sockets = cpu.get_sockets()

            set_allocation_class(sockets, class_id, mb_max)
            # print_allocation_config(sockets)
    except:
        print("Error!")
        raise



if __name__ == "__main__":
    main()
