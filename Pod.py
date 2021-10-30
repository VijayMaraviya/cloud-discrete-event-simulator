import random
import logging

# custom classes
from utils import Resource
from Node import Node

# setup logger
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(message)s')

file_handler = logging.FileHandler('Simulation.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class Pod:
    '''
    Base class for pod. 
    Subclass this and implement `process()` method to define pod behaviour.
    '''

    def __init__(self, env, name, resource_guarantee, resource_limit):
        '''
        `resource_limit` is a NamedTuple(memory, cpu) max limits.

        '''
        self._env = env
        self._name = name

        # pod resource gaurantees
        self._mem_guarantee = resource_guarantee.memory
        self._cpu_guarantee = resource_guarantee.cpu
        # pod resource limits
        self._mem_limit = resource_limit.memory
        self._cpu_limit = resource_limit.cpu

        # pod resource usage
        self._mem_usage = 0
        self._cpu_usage = 0
        # pod resource max usage
        self._max_mem_usage = 0
        self._max_cpu_usage = 0

        # pod attributes (set at runtime)
        self._create_time = None  # set by pod generator
        self._start_time = None  # set by pod `.process()`
        self._finish_time = None  # set by pod `.process()`
        self._success = None  # set by pod `.process()`
        # message when pod terminates (use to troublshoot pod failures)
        self._value = None  # set by pod `.process()`

        # Node assigned to pod
        self._node = None

    @property
    def name(self):
        return self._name

    @property
    def mem_guarantee(self):
        return self._mem_guarantee

    @property
    def cpu_guarantee(self):
        return self._cpu_guarantee

    @property
    def mem_limit(self):
        return self._mem_limit

    @property
    def cpu_limit(self):
        return self._cpu_limit

    @property
    def mem_usage(self):
        return self._mem_usage

    @property
    def cpu_usage(self):
        return self._cpu_usage

    @property
    def max_mem_usage(self):
        return self._max_mem_usage

    @property
    def max_cpu_usage(self):
        return self._max_cpu_usage

    @property
    def create_time(self):
        return self._create_time

    @create_time.setter
    def create_time(self, time):
        # TODO: check input before assignment
        if self._create_time is None:
            self._create_time = time
        else:
            raise ValueError('Pod already submitted.')

    @property
    def start_time(self):
        return self._start_time

    @property
    def finish_time(self):
        return self._finish_time

    @property
    def success(self):
        return self._success

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, node):
        if isinstance(node, Node):
            self._node = node
        else:
            raise ValueError("Invalid object type.")

    @property
    def value(self):
        return self._value

    def process(self, *args, **kwargs):
        '''
        Define the beahviour of a pod while running on an assigned node.

        '''
        raise NotImplementedError


class StaticPod(Pod):
    def __init__(self, env, name, resource_guarantee, resource_limit):
        super().__init__(env, name, resource_guarantee, resource_limit)

    def process(self, period_gen, resource_use_gen, schedular):
        '''
        `period_gen` is python generator which yields the time needed for pod to finish the job.
        `resource_use_gen` is python generator which yields the nametuple of resource used by the pod.
        `schedular` is the instance of Schedular Base class and allocates the node.
        '''
        # checks to ensure that one pod runs only once
        if self._success is not None:
            raise ValueError('Pod is already processed')

        try:

            with schedular.allocate(self) as res:
                # resume when node is allocated
                yield res

            # DEBUG: which node is asigned
            logger.debug(
                f"Time= {self._env.now} | {self.name} is assigned {self.node.ID}")

            # resources needed for pod to run
            mem_use, cpu_use = next(resource_use_gen)

            # acquire the resources from the assigned node
            with self._node.acquire(self, mem_use, cpu_use) as res:
                # resume when resources are obtained
                yield res

            # update pod attributes
            self._start_time = self._env.now
            self._mem_usage = mem_use
            self._cpu_usage = cpu_use
            self._max_mem_usage = mem_use
            self._max_cpu_usage = cpu_use

            # DEBUG: When and how much resources are obtained
            logger.debug(
                f"Time= {self._env.now} | {self.name} obtained RAM={mem_use} and CPU={cpu_use}")

            # generate the pod run time
            run_time = next(period_gen)
            # resume when pod has finished
            yield self._env.timeout(run_time)

            # release the resources back to the node
            with self._node.release(self, mem_use, cpu_use) as res:
                yield res

            self._mem_usage = 0
            self._cpu_usage = 0

            # update the finish time
            self._finish_time = self._env.now

            # DEBUG: When and how much resources are released
            logger.debug(
                f"Time= {self._env.now} | {self.name} released the resources and terminated")

            # mark success
            self._success = True

        except Exception as e:
            # DEBUG : When and why a pod is killed
            logger.debug(
                f"Time= {self._env.now} | {self.name} killed due to {e}")

            # mark failure
            self._success = False
            self._value = e
