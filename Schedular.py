import logging

# simpy classes
from simpy.events import Event
from simpy.core import BoundClass, Environment

# custom classes
from Pod import Resource
from utils import NodeMatchNotFound, PoolMaxCapacityError

# setup logger
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(message)s')

file_handler = logging.FileHandler('Simulation.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class Allocate(Event):
    ''' Event for requesting to allocate a node
    '''

    def __init__(self, schedular, pod):
        super().__init__(schedular._env)
        self.resource = schedular
        self.pod = pod
        self.proc = self.env.active_process

        schedular.put_queue.append(self)
        schedular._do_allocate(self)

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type,
                 exc_value,
                 traceback):
        self.cancel()
        return None

    def cancel(self):
        """Cancel this put request.

        This method has to be called if the put request must be aborted, for
        example if a process needs to handle an exception like an
        :class:`~simpy.exceptions.Interrupt`.

        If the put request was created in a :keyword:`with` statement, this
        method is called automatically.

        """
        if not self.triggered:
            self.resource.put_queue.remove(self)


class Schedular:

    PutQueue = list

    def __init__(self, env, name, cluster):
        self._env = env
        self._name = name
        self._cluster = cluster

        # request queues
        self.put_queue = self.PutQueue()

        # Bind event constructors as methods
        BoundClass.bind_early(self)

    # put a node into the pool
    allocate = BoundClass(Allocate)

    @property
    def name(self):
        return self._name

    @property
    def cluster(self):
        return self._cluster

    def _do_allocate(self, event):
        '''
        This method is called when a pod rquest to allocate (allocate event) the pod.
        Imlement the processing of allocation request.
        '''
        raise NotImplementedError


class BestFitBinPackingSchedular(Schedular):
    def __init__(self, env, name, cluster):
        super().__init__(env, name, cluster)

    def _do_allocate(self, event):
        """Perform the *put* operation.

        This method is called by :meth:`_trigger_put` for every event in the
        :attr:`put_queue`, as long as the return value does not evaluate
        ``False``.
        """

        demand = Resource(event.pod.mem_guarantee, event.pod.cpu_guarantee)
        try:
            assigned_node = self._cluster.get_node(
                demand, criteria='most_utilized')
            event.pod.node = assigned_node

            # succeed the request
            event._ok = True
            event._value = None
            # schedule the aloocation event immeadiately
            # TODO: logic is incorrect for delay start
            event.env.schedule(event, priority=0)
            # remove the event from the schedular queue
            self.put_queue.remove(event)

            # record the statistic
            # TODO: add record for statistics
            # self._node_metric.Record(self.node_count)

        except NodeMatchNotFound:

            self._env.process(self._attempt_to_scaleout(event, demand))

    def _attempt_to_scaleout(self, event, demand):
        # get nodepools and sort by size
        nodepools = sorted(self._cluster.pools,
                           key=lambda pool: pool.node_size)

        for nodepool in nodepools:
            try:
                # request to scaleout a pool
                with nodepool.scaleout() as req:
                    new_node = yield req

                # DEBUG: When and why a new node is added
                logger.debug(
                    f"Time= {self._env.now} | {new_node.ID} added while allocating {event.pod.name}")

                # # if request succeed, assign the new node to the pod
                # assigned_node = nodepool.get_least_utilized_node(demand)
                # TODO: check if works?
                event.pod.node = new_node

                # succeed the request
                event._ok = True
                event._value = None
                # schedule immeadiately
                # TODO: logic is incorrect for delay start
                event.env.schedule(event, priority=0)
                # remove the event from the schedular queue
                self.put_queue.remove(event)

                return None

            except PoolMaxCapacityError:

                # DEBUG
                logger.debug(
                    f"Time= {self._env.now} | failed to add a node to {nodepool.name}")

                # try with next nodepool
                pass

        # TODO: provide more informative error message
        # reject the request
        event._ok = False
        event._value = NodeMatchNotFound('Cannot allocate this pod!')
        event.env.schedule(event, 0)  # schedule immeadiately
        # remove the event from the schedular queue
        self.put_queue.remove(event)


class CustomSchedular(Schedular):
    def __init__(self, env, name, cluster):
        super().__init__(env, name, cluster)

    def _do_allocate(self, event):
        """Perform the *put* operation.

        This method is called by :meth:`_trigger_put` for every event in the
        :attr:`put_queue`, as long as the return value does not evaluate
        ``False``.
        """

        demand = Resource(event.pod.mem_guarantee, event.pod.cpu_guarantee)
        try:
            assigned_node = self._cluster.get_node(
                demand, criteria='most_utilized')
            event.pod.node = assigned_node

            # succeed the request
            event._ok = True
            event._value = None
            # schedule the aloocation event immeadiately
            # TODO: logic is incorrect for delay start
            event.env.schedule(event, priority=0)
            # remove the event from the schedular queue
            self.put_queue.remove(event)

            # record the statistic
            # TODO: add record for statistics
            # self._node_metric.Record(self.node_count)

        except NodeMatchNotFound:

            self._env.process(self._attempt_to_scaleout(event, demand))

    def _attempt_to_scaleout(self, event, demand):
        # get nodepools and sort by size
        nodepools = sorted(self._cluster.pools,
                           key=lambda pool: pool.node_size)

        for nodepool in nodepools:
            try:
                # request to scaleout a pool
                with nodepool.scaleout() as req:
                    new_node = yield req

                # DEBUG: When and why a new node is added
                logger.debug(
                    f"Time= {self._env.now} | {new_node.ID} added while allocating {event.pod.name}")

                # # if request succeed, assign the new node to the pod
                # assigned_node = nodepool.get_least_utilized_node(demand)
                # TODO: check if works?
                event.pod.node = new_node

                # succeed the request
                event._ok = True
                event._value = None
                # schedule immeadiately
                # TODO: logic is incorrect for delay start
                event.env.schedule(event, priority=0)
                # remove the event from the schedular queue
                self.put_queue.remove(event)

                return None

            except PoolMaxCapacityError:

                # DEBUG
                logger.debug(
                    f"Time= {self._env.now} | failed to add a node to {nodepool.name}")

                # try with next nodepool
                pass

        # TODO: provide more informative error message
        # reject the request
        event._ok = False
        event._value = NodeMatchNotFound('Cannot allocate this pod!')
        event.env.schedule(event, 0)  # schedule immeadiately
        # remove the event from the schedular queue
        self.put_queue.remove(event)
