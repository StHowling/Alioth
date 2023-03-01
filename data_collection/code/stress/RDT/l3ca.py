from pqos import Pqos
from pqos.cpuinfo import PqosCpuInfo
from pqos.l3ca import PqosCatL3

def set_allocation_class(sockets, class_id, mask):
    """
    Sets up allocation classes of service on selected CPU sockets

    Parameters:
        sockets: a list of socket IDs
        class_id: class of service ID
        mask: COS bitmask
    """

    l3ca = PqosCatL3()
    cos = l3ca.COS(class_id, mask)

    for socket in sockets:
        try:
            l3ca.set(socket, [cos])
            print("Socket %d Success!" %(socket))
        except:
            print("Setting up cache allocation class of service failed!")


def print_allocation_config(sockets):
    """
    Prints allocation configuration.

    Parameters:
        sockets: a list of socket IDs
    """

    l3ca = PqosCatL3()

    for socket in sockets:
        try:
            coses = l3ca.get(socket)

            print("L3CA COS definitions for Socket %u:" % socket)

            for cos in coses:
                cos_params = (cos.class_id, cos.mask)
                print("    L3CA COS%u => MASK 0x%x" % cos_params)
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
    cos_bitmask = 0x7ff
    try:
        with PqosContextManager("MSR"):
            cpuinfo = PqosCpuInfo()
            sockets = cpuinfo.get_sockets()

            set_allocation_class(sockets, 0, cos_bitmask)
            # print_allocation_config(sockets)
    except:
        print("Error!")
        raise

if __name__ == "__main__":
    main()