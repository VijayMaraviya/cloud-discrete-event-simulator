# %%
import random
import itertools
import os
import time
import numpy as np
import scipy.stats as sp

# custom classes
from Node import Node
from NodePool import NodePool
from Cluster import Cluster
from Schedular import BestFitBinPackingSchedular, CustomSchedular
from PodGenerator import PodGenerator
from utils import Resource, change_dir
import SimRNG

# simpy classes
import simpy.core as core

# %%

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.sem(a)
    h = se * sp.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def ArrivalGen():
    '''
    generate and yield interarrival time for pods

    '''
    # average time between two consective pod submission
    MeanTBA = 3/100

    while True:
        yield SimRNG.Expon(MeanTBA, Stream=10)


def PeriodGen():
    '''
    generate and yield pod life (activity period) 
    Erlang distribution
    '''
    # User spend (pod life is) on average 3 hours
    MeanAP = 3
    Phases = 3

    while True:
        yield SimRNG.Erlang(Phases, MeanAP, Stream=20)


rg1 = np.random.RandomState()
rg2 = np.random.RandomState()


def DemandGen():
    '''
    generate and yield pod resource demand
    '''
    while True:
        mem = rg1.uniform(512, 2048)
        cpu = rg2.uniform(0.05, 0.30)
        yield Resource(mem, cpu)


period_gen = PeriodGen()
resource_gen = DemandGen()
arrival_gen = ArrivalGen()

# %%

# node types (for cluster configuration)
node_types = {'E2s_v4': {'cost': 0.1767,
                         'size': Resource(16384, 2),
                         'reserved': Resource(3262.40, 0.20)},
              'E4s_v4': {'cost': 0.3533,
                         'size': Resource(32768, 4),
                         'reserved': Resource(4245.44, 0.24)},
              'E8s_v4': {'cost': 0.7066,
                         'size': Resource(65536, 8),
                         'reserved': Resource(6211.52, 0.28)},
              'E16s_v4': {'cost': 1.4132,
                          'size': Resource(131072, 16),
                          'reserved': Resource(10143.68, 0.36)},
              'E20s_v4': {'cost': 1.7664,
                          'size': Resource(163840, 20),
                          'reserved': Resource(10801.60, 0.42)},
              'E32s_v4': {'cost': 2.8263,
                          'size': Resource(262144, 32),
                          'reserved': Resource(12765.12, 0.52)},
              'E48s_v4': {'cost': 4.2394,
                          'size': Resource(393216, 48),
                          'reserved': Resource(15386.56, 0.67)},
              'E64s_v4': {'cost': 5.6525,
                          'size': Resource(516096, 64),
                          'reserved': Resource(17844.16, 0.84)}
              }

# %%
type = 'E2s_E20s_v4'
config1 = node_types['E20s_v4']
config2 = node_types['E2s_v4']


# use common random number across scenarios
SimRNG.ZRNG = SimRNG.InitializeRNSeed()
rg1.seed(1)
rg2.seed(2)

all_success_prob = []
all_cluster_cost = []

# run 10 times
for i in range(10):

    env = core.Environment()

    # config for node pool
    pool_A = NodePool(env, 'E20s_v4',
                      node_size=config1['size'], node_cost=config1['cost'],
                      min_count=1, max_count=100, node_init=config1['reserved'])

    # config for node pool
    pool_B = NodePool(env, 'E2s_v4',
                      node_size=config2['size'], node_cost=config2['cost'],
                      min_count=0, max_count=2, node_init=config2['reserved'])

    # list of pools in cluster
    pools = [pool_A, pool_B]

    # cluster
    cluster = Cluster(env, 'cluster' + type, pools)

    # schedular
    schedular = CustomSchedular(env, 'schedular', cluster)

    # pod attributes (limit and guarantee)
    resource_guarantee = Resource(512, 0.05)
    resource_limit = Resource(2048, 0.30)
    # pod generator
    pod_generator = PodGenerator(env, 'Static Pod Generator', arrival_gen)
    env.process(pod_generator.generate(resource_guarantee,
                                       resource_limit,
                                       period_gen,
                                       resource_gen,
                                       schedular
                                       )
                )

    # status update on consol
    print(f'running simulation for config type: {type}')

    # run simulation
    env.run(until=100)

    # get results
    cluster_res = cluster.get_results()
    pod_gen_res = pod_generator.get_results()

    # avg cluster cost
    avg_cost = cluster_res['cluster' + type].cost
    # success probabaility
    success_prob = pod_gen_res['Static Pod Generator'].success_probability

    all_cluster_cost.append(avg_cost)
    all_success_prob.append(success_prob)

m, lb, ub = mean_confidence_interval(all_cluster_cost)
print(
    f'estimated expected unit cost for config type{type} = {m:.3f} $ and 95% CI of estimate is [{lb:.3f}, {ub:.3f}]')

m, lb, ub = mean_confidence_interval(all_success_prob)
print(
    f'estimated expected success probability for config type{type} = {m:.3f} and 95% CI of estimate is [{lb:.3f}, {ub:.3f}]')


# %%
