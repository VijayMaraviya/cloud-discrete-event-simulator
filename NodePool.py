from heapq import heappush, heappop, nlargest, heapify
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
import csv
import os
import logging

# custom classes
from SimStatistics import CTStat, DTStat
from Node import Node
from utils import (PoolMaxCapacityError,
                   PoolMinCapacityError,
                   NodePodCountNotZero,
                   NodeMatchNotFound,
                   change_dir)

# simpy classes
from simpy.core import BoundClass, Environment
from simpy.resources import base
from simpy.events import Event


# setup logger
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(message)s')

file_handler = logging.FileHandler('Simulation.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

NodeSize = namedtuple('NodeSize', ['memory', 'cpu'])


class ScaleOut(Event):
    ''' Event for requesting to add a node to the pool
    '''

    def __init__(self, nodepool):
        super().__init__(nodepool._env)
        self.resource = nodepool
        self.proc = self.env.active_process

        nodepool.put_queue.append(self)
        nodepool._trigger_put(None)

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


class ScaleIn(Event):
    ''' Event for requesting to remove a node from the pool
    '''

    def __init__(self, nodepool):

        super().__init__(nodepool._env)
        self.resource = nodepool
        self.proc = self.env.active_process

        nodepool.get_queue.append(self)
        nodepool._trigger_get(None)

    def __enter__(self):
        return self

    def __exit__(
            self,
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
            self.resource.get_queue.remove(self)


class NodePool:
    ''' Pool containing nodes of same type between `min_count` and `max_count`.
    It supports the request to `scalein` and `scaleout` nodes from/into the pool.
    '''
    PutQueue = list
    GetQueue = list

    def __init__(self, env, name, node_size, node_cost, min_count, max_count, node_init, start_delay=0):
        '''
        node_size is a NamedTuple(memory, cpu).
        node_init is a NamedTuple(memory, cpu) to consume from Node at all times. (overhead)
        '''
        self._env = env
        self._name = name

        # delay in start (ready for use) of a new node after bootup
        self._node_start_delay = start_delay

        # node type
        self._node_memory = node_size.memory
        self._node_cpu = node_size.cpu

        # node cost (per unit time)
        self._c = node_cost

        # node ovehead
        self._node_mem_init = node_init.memory
        self._node_cpu_init = node_init.cpu

        # auto scalling limits
        self._min_count = min_count
        self._max_count = max_count

        # list of active nodes in the pool
        self._nodes = []
        self._nid = count(1)  # for node id

        # list of all nodes (active and inactive)
        self._all_nodes = []

        # request queues
        self.put_queue = self.PutQueue()
        self.get_queue = self.GetQueue()

        # Bind event constructors as methods
        BoundClass.bind_early(self)

        # statistics
        # node count over time
        self._node_metric = CTStat(env, env.now, min_count)
        self._node_results = None
        # cost (avg per unit time for this pool)
        self._cost = None

        # add nodes to the pool
        for _ in range(min_count):
            self.scaleout()

    @property
    def name(self):
        return self._name

    @property
    def node_count(self):
        return len(self._nodes)

    @property
    def node_min_count(self):
        return self._min_count

    @property
    def node_max_count(self):
        return self._max_count

    @property
    def node_size(self):
        return NodeSize(self._node_memory, self._node_cpu)

    @property
    def node_cost(self):
        return self._c

    @property
    def pool_cost(self):
        return self._cost

    @property
    def nodes(self):
        return self._nodes

    @property
    def pod_count(self):
        count = 0
        for node in self._nodes:
            count += node.pod_count
        return count

    @property
    def pods(self):
        pods = []
        for node in self._nodes:
            pods.extend(node.pods)
        return pods

    # put a node into the pool
    scaleout = BoundClass(ScaleOut)
    # get a node out of the node
    scalein = BoundClass(ScaleIn)

    def heap_update(self):
        '''
        update heap after any chnages to the node state.
        Called by node after state change.
        '''
        heapify(self._nodes)

    def get_most_utilized_node(self, demand=None):
        '''
        returns the node being most utilized 
        scheduler call this method.
        'demand' is a NamedTuple(memory, cpu).
        '''
        # create a shallow copy of `self._nodes`
        nodes = self._nodes.copy()
        for _ in range(len(nodes)):
            candidate = heappop(nodes)
            if demand is None:
                return candidate
            elif (candidate.memory_avail > demand.memory) and (candidate.CPU_avail > demand.cpu):
                return candidate

        raise NodeMatchNotFound(
            f'None of the nodes on {self._name} can match the demand')

    def get_least_utilized_node(self, demand=None):
        '''
        returns the node being most utilized 
        scheduler call this method.
        'demand' is a NamedTuple(memory, cpu).
        '''
        # get the node with the highest capacity
        candidate = nlargest(1, self._nodes)
        if demand is None:
            return candidate[0]
        elif (candidate[0].memory_avail > demand.memory) and (candidate[0].CPU_avail > demand.cpu):
            return candidate[0]

        raise NodeMatchNotFound(
            f'None of the nodes on {self._name} can match the demand')

    def _do_put(self, event):
        """Perform the *put* operation.

        This method is called by :meth:`_trigger_put` for every event in the
        :attr:`put_queue`, as long as the return value does not evaluate
        ``False``.
        """
        if self.node_count < self._max_count:
            # Instantiate a new node
            # TODO: change node name
            new_node = Node(self._env, self._name + '_Node_' + str(next(self._nid)), self._node_cpu,
                            self._node_memory, self._node_cpu_init, self._node_mem_init, self)

            # add a new node to the pool
            heappush(self._nodes, new_node)

            # update all_nodes list
            self._all_nodes.append(new_node)

            # succeed the request
            event._ok = True
            # TODO: check if works?
            event._value = new_node
            # schedule immeadiately
            # TODO: logic is incorrect for delay start
            event.env.schedule(event, priority=0, delay=self._node_start_delay)

            # record the statistic
            # TODO: add record for statistics
            self.record()

            return True

        else:
            # TODO: Handle the event manually
            event.fail(PoolMaxCapacityError(
                'Node count cannot exceed node_max_count!'))
            return None

    def _trigger_put(self, get_event):
        """This method is called once a new put(release) event has been created or a get(acquire)
        event has been processed.

        The method iterates over all put events in the :attr:`put_queue` and
        calls :meth:`_do_put` to check if the conditions for the event are met.
        If :meth:`_do_put` returns ``False``, the iteration is stopped early.
        """
        idx = 0
        while idx < len(self.put_queue):
            put_event = self.put_queue[idx]
            proceed = self._do_put(put_event)
            if not put_event.triggered:
                idx += 1
            elif self.put_queue.pop(idx) != put_event:
                raise RuntimeError('Put queue invariant violated')

            if not proceed:
                break

    def _do_get(self, event):
        """Perform the *get* operation.

        This method is called by :meth:`_trigger_get` for every event in the
        :attr:`get_queue`, as long as the return value does not evaluate
        ``False``.
        """
        if self.node_count > self._min_count:
            # most empty node (node with max available capacity)
            max_node = nlargest(1, self._nodes)[0]

            # check if pod count of a node with max capacity is 0
            if max_node.pod_count == 0:
                # record the node statistics
                max_node.record()
                # shutdown the node
                max_node.shutdown_time = self._env.now

                # remove node from the active nodes list
                self._nodes.remove(max_node)
                heapify(self._nodes)

                # DEBUG: print result and node count
                logger.debug(
                    f"Time= {self._env.now} | {max_node.ID} is removed (new node count={self.node_count})")

                # succeed the request
                event._ok = True
                event._value = None
                # schedule immeadiately
                event.env.schedule(event, priority=0, delay=0)

                # record the pool statistic
                self.record()

                return True

            else:
                # don't raise runtime error
                event._defused = True
                event._ok = False
                event._value = NodePodCountNotZero(
                    'All nodes have at least 1 pod running!')
                event.env.schedule(event, 0)  # schedule immeadiately
                return None

        else:
            # don't raise runtime error
            event._defused = True
            event._ok = False
            event._value = PoolMinCapacityError(
                'Node count cannot reduce below node_min_count!')
            event.env.schedule(event, 0)  # schedule immeadiately
            return None

    def _trigger_get(self, put_event):
        """Trigger get events.

        This method is called once a new get event has been created or a put
        event has been processed.

        The method iterates over all get events in the :attr:`get_queue` and
        calls :meth:`_do_get` to check if the conditions for the event are met.
        If :meth:`_do_get` returns ``False``, the iteration is stopped early.
        """
        idx = 0
        while idx < len(self.get_queue):
            get_event = self.get_queue[idx]
            proceed = self._do_get(get_event)
            if not get_event.triggered:
                idx += 1
            elif self.get_queue.pop(idx) != get_event:
                raise RuntimeError('Get queue invariant violated')

            if not proceed:
                break

    def record(self):
        # record the pool statistic
        self._node_metric.Record(self.node_count)

        # calculate results
        UsageResult = namedtuple('UsageResult', ['average', 'max', 'min'])

        # node count (upto current time)
        avg_node_count = self._node_metric.Mean()
        max_node_count = self._node_metric.Max()
        min_node_count = self._node_metric.Min()
        self._node_results = UsageResult(avg_node_count,
                                         max_node_count,
                                         min_node_count)

        # cost (per unit time)(averaged upto current time)
        self._cost = self._c*avg_node_count

    def clear(self):
        self._node_metric.Clear()

        self.record()

    def get_results(self):
        '''
        returns a dictionary of results data
        '''
        # record the statistics upto current time
        self.record()

        PoolResult = namedtuple('PoolResults', ['cost', 'node_count', 'nodes'])

        # get the metrics of each node inside the pool
        nodes_metrics = {}
        for node in self._all_nodes:
            nodes_metrics.update(node.get_results())

        # all pool level results
        results = PoolResult(self._cost, self._node_results, nodes_metrics)

        return {self._name: results}

    def save_results(self, destination=None):
        # record the statistics
        self._node_metric.Record(self.node_count)

        if destination is None:
            destination = os.getcwd()

        # inside destination directory
        with change_dir(destination):
            pool_dir = self._name
            # create pool directory
            os.mkdir(pool_dir)

            # inside pool directory
            with change_dir(pool_dir):

                # data (metrics)
                sim_time = self._node_metric.T_list
                node_count = self._node_metric.X_list
                node_AUC = self._node_metric.A_list
                avg_node_count = self._node_metric.t_avg_list

                row_list = zip(sim_time, node_count, node_AUC, avg_node_count)

                target_file = self.name + '-Metrics.csv'

                # create a csv file
                with open(target_file, 'w', newline='') as new_file:

                    fieldnames = ['sim_time', 'node_count',
                                  'node_AUC', 'avg_node_count']

                    csv_writer = csv.writer(new_file)

                    csv_writer.writerow(fieldnames)
                    for row in row_list:
                        csv_writer.writerow(row)

                # create plots directory
                os.mkdir('plots')

                # inside plots directory
                with change_dir('plots'):
                    # save plots
                    self._plot_and_save(sim_time, node_count,
                                        'simulation time', 'node count')
                    self._plot_and_save(sim_time, avg_node_count,
                                        'simulation time', 'average node count', where='pre')

            # save node results
            for node in self._all_nodes:
                node.save_results(pool_dir)

    def _plot_and_save(self, x, y, xlabel, ylabel, destination=None, where='post'):
        if destination is None:
            destination = os.getcwd()

        # new figure and associated axes
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot()

        # plot
        ax1.step(x, y, where=where)

        # set labels and titles
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(f'{ylabel} vs {xlabel} for {self.name}')

        # save file at destination directory
        fig1.savefig(os.path.join(
            destination, xlabel + '_vs_' + ylabel + '.png'))

        # close the figure
        plt.close(fig1)
