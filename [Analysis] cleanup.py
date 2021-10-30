# %%
import random
import itertools
import os
import time
import numpy as np
import scipy.stats as sp
import logging

# custom classes
from Node import Node
from NodePool import NodePool
from Cluster import Cluster
from Schedular import BestFitBinPackingSchedular
from PodGenerator import PodGenerator
from utils import Resource, change_dir
import SimRNG

# simpy classes
import simpy.core as core

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(message)s')
file_handler = logging.FileHandler('simulation.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
node_types = {'E16s_v4': {'cost': 1.4132,
                          'size': Resource(131072, 16),
                          'reserved': Resource(10143.68, 0.36)},
              'E20s_v4': {'cost': 1.7664,
                          'size': Resource(163840, 20),
                          'reserved': Resource(10801.60, 0.42)}}

# %%

# for all scenarios run simulation
for type, config in node_types.items():

    all_success_prob = []
    all_cluster_cost = []

    # use common random number across scenarios
    SimRNG.ZRNG = SimRNG.InitializeRNSeed()
    rg1.seed(1)
    rg2.seed(2)

    # run 10 times
    for i in range(10):

        env = core.Environment()

        # config for node pool
        pool_A = NodePool(env, type,
                          node_size=config['size'], node_cost=config['cost'],
                          min_count=1, max_count=100, node_init=config['reserved'])

        # list of pools in cluster
        pools = [pool_A]

        # cluster
        cluster = Cluster(env, 'cluster' + type, pools)

        # schedular
        schedular = BestFitBinPackingSchedular(env, 'schedular', cluster)

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
        env.run(until=500)

        # get results
        cluster_res = cluster.get_results()
        pod_gen_res = pod_generator.get_results()

        # avg cluster cost
        avg_cost = cluster_res['cluster' + type].cost
        # success probabaility
        success_prob = pod_gen_res['Static Pod Generator'].success_probability

        all_cluster_cost.append(avg_cost)
        all_success_prob.append(success_prob)

        # # print
        # print(f"run: {i}, Cost for config type {type}={avg_cost:.3f}")
        # print(
        #     f"run: {i}, success probability for config type {type}={success_prob:.3f}")

    m, lb, ub = mean_confidence_interval(all_cluster_cost)
    print(
        f'estimated expected unit cost for config type{type} = {m:.3f} $ and 95% CI of estimate is [{lb:.3f}, {ub:.3f}]')

    m, lb, ub = mean_confidence_interval(all_success_prob)
    print(
        f'estimated expected success probability for config type{type} = {m:.3f} and 95% CI of estimate is [{lb:.3f}, {ub:.3f}]')


# %%
