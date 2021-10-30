from collections import namedtuple
from contextlib import contextmanager
import os
from simpy.exceptions import SimPyException

Resource = namedtuple('Resource', ['memory', 'cpu'])


class PoolMaxCapacityError(SimPyException):
    '''
    Pool cannot add more nodes than 'node_max_count'.
    '''

    def __init__(self, msg):
        super().__init__(msg)


class PoolMinCapacityError(SimPyException):
    '''
    Pool cannot remove nodes below 'node_min_count'.
    '''

    def __init__(self, msg):
        super().__init__(msg)


class NodePodCountNotZero(SimPyException):
    '''
    Pool cannot remove a node if pods are running on it.
    '''

    def __init__(self, msg):
        super().__init__(msg)


class NodeMatchNotFound(SimPyException):
    '''
    Pool cannot find a node that can fulfill the requested demand.
    '''

    def __init__(self, msg):
        super().__init__(msg)


class NodeOOMError(SimPyException):
    '''
    Node do not have enough memeory.
    '''

    def __init__(self, msg):
        super().__init__(msg)


class NodeCPUError(SimPyException):
    '''
    Node do not have enough CPU.
    '''

    def __init__(self, msg):
        super().__init__(msg)


@contextmanager
def change_dir(destination):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)
