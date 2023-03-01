from pqos import Pqos
from pqos.allocation import PqosAlloc
from pqos.cpuinfo import PqosCpuInfo

def set_allocation_class(class_id, cores):
    """
    Sets up allocation classes of service on selected CPUs

    Parameters:
        class_id: class of service ID
        cores: a list of cores
    """

    alloc = PqosAlloc()

    for core in cores:
        try:
            alloc.assoc_set(core, class_id)
            print("Success!")
        except:
            print("Setting allocation class of service association failed!")

def reset_allocation():
    """
    Resets allocation configuration.
    """

    alloc = PqosAlloc()

    try:
        alloc.reset('any', 'any', 'any')
        print("Allocation reset successful")
    except:
        print("Allocation reset failed!")


def print_allocation(core_list):
    alloc = PqosAlloc()
    try:
        for core in core_list:
            class_id = alloc.assoc_get(core)
            try:
                print("\tCore %u => COS%u" % (core, class_id))
            except:
                print("\tCore %u => ERROR" % (core))
    except:
        print("print Error")
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
    core_list = [0]
    class_id = 8
    try:
        with PqosContextManager("MSR"):
            if core_list:
                set_allocation_class(class_id, core_list)
            print_allocation(core_list)
            # reset_allocation()
    except:
        print("Error!")
        raise

if __name__ == "__main__":
    main()
