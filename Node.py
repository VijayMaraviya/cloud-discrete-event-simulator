import os
import csv
import matplotlib.pyplot as plt
from collections import namedtuple

# custom classes
from SimStatistics import CTStat, DTStat
from utils import (PoolMinCapacityError,
                   NodeOOMError,
                   NodeCPUError,
                   change_dir)

# simpy classes
from simpy.core import BoundClass, Environment
from simpy.resources import base
from simpy.exceptions import SimPyException


class NodeRelease(base.Put):
    '''Event for requesting to release compute resources

    Raise a `ValueError` if ``resources <= 0``.
    '''

    def __init__(self, node, pod, memory, CPU):
        if CPU < 0:
            raise ValueError(f'CPU(={CPU}) must be >=0.')
        if memory < 0:
            raise ValueError(f'memory(={memory}) must be >=0.')

        self.pod = pod

        self.CPU = CPU
        self.memory = memory

        super().__init__(node)


class NodeAcquire(base.Get):
    '''Event for requesting to aquire compute resources

    Raise a `ValueError` if ``resources <= 0``.
    '''

    def __init__(self, node, pod, memory, CPU):
        if CPU < 0:
            raise ValueError(f'CPU(={CPU}) must be >=0.')
        if memory < 0:
            raise ValueError(f'memory(={memory}) must be >=0.')

        self.pod = pod

        self.CPU = CPU
        self.memory = memory

        super().__init__(node)


class Node:
    ''' Node containing up to capacity of compute resources.
    It supports the request to acquire and release resources from/into the Node.

    Raise a `ValueError` if resource capacity <= 0, init < 0, or init > capacity
    '''

    PutQueue = list
    GetQueue = list

    def __init__(
            self,
            env,
            Node_ID,
            CPU_capacity,
            memory_capacity,
            CPU_init=0,
            memory_init=0,
            pool=None,):
        if CPU_capacity <= 0:
            raise ValueError('"CPU_capacity" must be > 0.')
        if CPU_init < 0:
            raise ValueError('"CPU_init" must be >= 0.')
        if CPU_init > CPU_capacity:
            raise ValueError('"CPU_init" must be <= "CPU_capacity".')
        if memory_capacity <= 0:
            raise ValueError('"memory_capacity" must be > 0.')
        if memory_init < 0:
            raise ValueError('"memory_init" must be >= 0.')
        if memory_init > memory_capacity:
            raise ValueError('"memory_init" must be <= "memory_capacity".')

        # unique id to identify the node (provided by NodePool)
        self._Node_ID = Node_ID
        # nodepool to which node belong, if any
        self._nodepool = pool
        self._env = env
        self._bootup_time = env.now
        self._shutdown_time = float('inf')
        self._pods = set()

        # Node capacity
        self._CPU_capacity = CPU_capacity
        self._memory_capacity = memory_capacity

        # Available CPU and memory
        self._CPU_avail = CPU_capacity - CPU_init
        self._memory_avail = memory_capacity - memory_init

        # request queues
        self.put_queue = self.PutQueue()
        self.get_queue = self.GetQueue()

        # Bind event constructors as methods
        BoundClass.bind_early(self)

        # statistics (Node Metrics)
        # memory usage over time
        self._mem_metric = CTStat(env, env.now, memory_init)
        # cpu usage over time
        self._cpu_metric = CTStat(env, env.now, CPU_init)
        # pod count over time
        self._pod_metric = CTStat(env, env.now, 0)

        # result (summary statistics)
        self._mem_results = None
        self._cpu_results = None
        self._pod_results = None

    @property
    def ID(self):
        return self._Node_ID

    @property
    def nodepool(self):
        return self._nodepool

    @property
    def bootup_time(self):
        return self._bootup_time

    @property
    def shutdown_time(self):
        return self._shutdown_time

    @shutdown_time.setter
    def shutdown_time(self, time):
        '''
        shutdown time is set by NodePool
        '''
        if time >= self._env.now:
            self._shutdown_time = time
        else:
            raise ValueError("Invalid time")

    @property
    def pod_count(self):
        return len(self._pods)

    @property
    def pods(self):
        return self._pods

    @property
    def CPU_avail(self):
        """The current available CPU in the node."""
        return self._CPU_avail

    @property
    def memory_avail(self):
        """The current available memeroy in the node."""
        return self._memory_avail

    @property
    def CPU_capacity(self):
        """Maximum CPU capacity of the node."""
        return self._CPU_capacity

    @property
    def memory_capacity(self):
        """Maximum memory capacity of the node."""
        return self._memory_capacity

    # put resources into the node
    release = BoundClass(NodeRelease)
    # get resources from the node
    acquire = BoundClass(NodeAcquire)

    def __lt__(self, other):
        return (self.memory_avail, self.CPU_avail, self.bootup_time) < (other.memory_avail, other.CPU_avail, other.bootup_time)

    def _nodepool_heap_update(self):
        '''
        update the position of node in nodepool heap after any state change
        '''
        self._nodepool.heap_update()

    def _do_put(self, event):
        """Perform the *put* operation.

        This method is called by :meth:`_trigger_put` for every event in the
        :attr:`put_queue`, as long as the return value does not evaluate
        ``False``.
        """
        if (self._CPU_capacity - self._CPU_avail >= event.CPU) and (self._memory_capacity - self._memory_avail >= event.memory):
            # add the resources back
            self._CPU_avail += event.CPU
            self._memory_avail += event.memory

            if event.pod.mem_usage - event.memory == 0:
                # remove requester pod from pods list
                self._pods.remove(event.pod)

                # (autoscalling) if pod_count is zero, ask pool to scale in
                if self.pod_count == 0:
                    try:
                        self._nodepool.scalein()
                    except PoolMinCapacityError:
                        pass

            # update node position in the nodepool heap
            if self._nodepool is not None:
                self._nodepool_heap_update()

            # succeed the request
            # event.succeed()
            event._ok = True
            event._value = None
            event.env.schedule(event, 0)  # schedule immeadiately

            # record the node statistic
            self.record()

            return True

        else:
            # TODO: should raise an exception
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
        if self._CPU_avail >= event.CPU and self._memory_avail >= event.memory:
            # allocated requested resources
            self._CPU_avail -= event.CPU
            self._memory_avail -= event.memory

            # add the requster pod to the pods list
            self._pods.add(event.pod)

            # update node position in the nodepool heap
            if self._nodepool is not None:
                self._nodepool_heap_update()

            # succeed the request
            # event.succeed()
            event._ok = True
            event._value = None
            event.env.schedule(event, 0)  # schedule immeadiately

            # record the statistic
            self._mem_metric.Record(self._memory_capacity - self._memory_avail)
            self._cpu_metric.Record(self._CPU_capacity - self._CPU_avail)
            self._pod_metric.Record(self.pod_count)

            return True

        elif self._CPU_avail < event.CPU:
            # reject the request
            event._ok = False
            event._value = NodeCPUError('Node CPU Error!')
            event.env.schedule(event, 0)  # schedule immeadiately
            return None

        elif self._memory_avail < event.memory:
            # reject the request
            event._ok = False
            event._value = NodeOOMError('Node Memory Error!')
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
        # if node is running
        if self._shutdown_time == float('inf'):
            # record statistics
            self._mem_metric.Record(self._memory_capacity - self._memory_avail)
            self._cpu_metric.Record(self._CPU_capacity - self._CPU_avail)
            self._pod_metric.Record(self.pod_count)

            # TODO: what results should be reported?
            # calculate results
            UsageResult = namedtuple('UsageResult', ['average', 'max', 'min'])
            # memory
            avg_mem_usage = self._mem_metric.Mean()
            max_mem_usage = self._mem_metric.Max()
            min_mem_usage = self._mem_metric.Min()
            self._mem_results = UsageResult(avg_mem_usage,
                                            max_mem_usage,
                                            min_mem_usage)
            # cpu
            avg_cpu_usage = self._cpu_metric.Mean()
            max_cpu_usage = self._cpu_metric.Max()
            min_cpu_usage = self._cpu_metric.Min()
            self._cpu_results = UsageResult(avg_cpu_usage,
                                            max_cpu_usage,
                                            min_cpu_usage)

            # pod count
            avg_pod_count = self._pod_metric.Mean()
            max_pod_count = self._pod_metric.Max()
            min_pod_count = self._pod_metric.Min()
            self._pod_results = UsageResult(avg_pod_count,
                                            max_pod_count,
                                            min_pod_count)

    def clear(self):
        # memory
        self._mem_metric.Clear()

        # cpu
        self._cpu_metric.Clear()

        # pod count
        self._pod_metric.Clear()

        self.record()

    def get_results(self):
        '''
        returns a dictionary of results data
        '''
        # record the statistics upto current time
        self.record()

        NodeResult = namedtuple('NodeResults', ['memory', 'cpu', 'pod_count'])

        results = NodeResult(
            self._mem_results, self._cpu_results, self._pod_results)

        return {self.ID: results}

    def save_results(self, destination=None):

        # record the node statistics
        self.record()

        if destination is None:
            destination = os.getcwd()

        # inside destination directory
        with change_dir(destination):
            node_dir = self._Node_ID
            # create a node directory
            os.mkdir(node_dir)

            # inside a node directory
            with change_dir(node_dir):

                # TODO: add avergae usgae
                # data (metrics)
                sim_time = self._mem_metric.T_list
                mem_usage = self._mem_metric.X_list
                mem_AUC = self._mem_metric.A_list
                cpu_usage = self._cpu_metric.X_list
                cpu_AUC = self._cpu_metric.A_list
                pod_count = self._pod_metric.X_list
                pod_AUC = self._pod_metric.A_list

                row_list = zip(sim_time, mem_usage, mem_AUC,
                               cpu_usage, cpu_AUC, pod_count, pod_AUC)

                target_file = self.ID + '-Metrics.csv'

                # create a csv file
                with open(target_file, 'w', newline='') as new_file:

                    fieldnames = ['sim_time', 'memory_usage', 'memory_AUC',
                                  'cpu_usage', 'cpu_AUC', 'pod_count', 'pod_AUC']

                    csv_writer = csv.writer(new_file)

                    csv_writer.writerow(fieldnames)
                    for row in row_list:
                        csv_writer.writerow(row)

                # create plots directory
                os.mkdir('plots')

                # inside plots directory
                with change_dir('plots'):
                    # save plots
                    self._plot_and_save(sim_time, mem_usage,
                                        'simulation time', 'memory usage')
                    self._plot_and_save(sim_time, cpu_usage,
                                        'simulation time', 'cpu usage')
                    self._plot_and_save(sim_time, pod_count,
                                        'simulation time', 'pod count')

    def _plot_and_save(self, x, y, xlabel, ylabel, destination=None):
        if destination is None:
            destination = os.getcwd()

        # new figure and associated axes
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot()

        # plot
        ax1.step(x, y, where='post')

        # set labels and titles
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(f'total {ylabel} vs {xlabel} for {self.ID}')

        # save file at destination directory
        fig1.savefig(os.path.join(destination, ylabel + '.png'))

        # close the figure
        plt.close(fig1)
