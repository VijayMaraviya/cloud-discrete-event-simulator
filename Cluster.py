import random
import os
import time
from collections import namedtuple
import logging

# custom classes
from utils import NodeMatchNotFound, change_dir


# setup logger
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(message)s')

file_handler = logging.FileHandler('Simulation.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class Cluster:
    '''
    Cluster is a group of nodepools. 
    A nodepool contains nodes of same type and can scale up and down.
    '''

    def __init__(self, env, name, pools, seed=42):
        self._env = env
        self._name = name
        self._pools = pools

    @ property
    def name(self):
        return self._name

    @property
    def pools(self):
        '''
        returns the list of nodepools in the cluster
        '''
        return self._pools

    @property
    def cost(self):
        '''
        total cluster cost (per unit time)(averaged upto current time)
        '''
        cost = 0
        for pool in self._pools:
            cost += pool.pool_cost
        return cost

    @property
    def nodes(self):
        '''
        returns the list of active nodes in the cluster
        '''
        _nodes = []
        for pool in self._pools:
            _nodes.extend(pool.nodes)
        return _nodes

    @property
    def node_count(self):
        '''
        returns the count of total active nodes in the cluster
        '''
        count = 0
        for pool in self._pools:
            count += pool.node_count
        return count

    @property
    def pods(self):
        '''
        returns the list of running pods in the cluster
        '''
        pods = []
        for pool in self._pools:
            pods.extend(pool.pods)
        return pods

    @property
    def pod_count(self):
        '''
        returns the count of total running pods in the cluster
        '''
        count = 0
        for pool in self._pools:
            count += pool.pod_count
        return count

    def get_node(self, demand=None, criteria='most_utilized'):
        '''
        returns a node that match the demand based on criteria.

        '''
        # TODO: check demand type
        criteria_list = ['most_utilized', 'least_utilized',
                         'most_pods', 'least_pods', 'random']
        if criteria not in criteria_list:
            raise ValueError(
                'Invalid rule. Expected one of: %s' % criteria_list)

        if criteria == 'random':
            return random.choice(self.nodes)

        # sort based on memory_avail, cpu_avail, and bootup_time
        nodes = sorted(self.nodes)

        if criteria == 'most_utilized':
            if demand is None:
                return nodes[0]
            else:
                for candidate in nodes:
                    if (candidate.memory_avail > demand.memory) and (candidate.CPU_avail > demand.cpu):
                        return candidate

            # no suitable node found
            # TODO: should I use exception here?
            raise NodeMatchNotFound(
                f'None of the nodes on {self._name} can match the demand')

        if criteria == 'least_utilized':
            if demand is None:
                return nodes[-1]
            else:
                for candidate in reversed(nodes):
                    if (candidate.memory_avail > demand.memory) and (candidate.CPU_avail > demand.cpu):
                        return candidate

            # no suitable node found
            # TODO: should I use exception here?
            raise NodeMatchNotFound(
                f'None of the nodes on {self._name} can match the demand')

        # sort nodes based on pod count
        nodes = sorted(nodes, key=lambda node: node.pod_count)

        if criteria == 'most_pods':
            if demand is None:
                return nodes[-1]
            else:
                for candidate in reversed(nodes):
                    if (candidate.memory_avail > demand.memory) and (candidate.CPU_avail > demand.cpu):
                        return candidate

            # no suitable node found
            # TODO: should I use exception here?
            raise NodeMatchNotFound(
                f'None of the nodes on {self._name} can match the demand')

        if criteria == 'least_pods':
            if demand is None:
                return nodes[0]
            else:
                for candidate in nodes:
                    if (candidate.memory_avail > demand.memory) and (candidate.CPU_avail > demand.cpu):
                        return candidate

            # no suitable node found
            # TODO: should I use exception here?
            raise NodeMatchNotFound(
                f'None of the nodes on {self._name} can match the demand')

    def get_noodpool(self, criteria='most_pods'):
        '''
        returns a noodepool based on criteria
        '''
        criteria_list = ['most_pods', 'least_pods',
                         'max_size', 'min_size'
                         'most_nodes', 'least_nodes', 'random']

        if criteria not in criteria_list:
            raise ValueError(
                'Invalid rule. Expected one of: %s' % criteria_list)

        if criteria == 'random':
            return random.choice(self._pools)

        # sort nodepools based on node size
        pools = sorted(self._pools,
                       key=lambda pool: (pool.node_memory, pool.node_cpu))

        if criteria == 'min_size':
            return pools[0]

        if criteria == 'max_size':
            return pools[-1]

        # sort nodepools based on node count
        pools = sorted(self._pools,
                       key=lambda pool: pool.node_count)

        if criteria == 'least_nodes':
            return pools[0]

        if criteria == 'most_nodes':
            return pools[-1]

        # sort nodepools based on pod count
        pools = sorted(self._pools,
                       key=lambda pool: pool.pod_count)

        if criteria == 'least_pods':
            return pools[0]

        if criteria == 'most_pods':
            return pools[-1]

    def get_results(self):
        '''
        returns a dictionary of cluster level results
        '''
        ClusterResult = namedtuple('ClusterResults', ['cost', 'pools'])

        # get the metrics of each pool inside the cluster
        pools_metrics = {}
        for pool in self._pools:
            pools_metrics.update(pool.get_results())

        # all pool level results
        results = ClusterResult(self.cost, pools_metrics)

        return {self._name: results}

    def save_results(self, destination=None):
        if destination is None:
            destination = os.getcwd()

        # inside destination directory
        with change_dir(destination):
            cluster_dir = self._name
            # create a cluster directory
            os.mkdir(cluster_dir)

            # save pool results
            for pool in self._pools:
                pool.save_results(cluster_dir)
