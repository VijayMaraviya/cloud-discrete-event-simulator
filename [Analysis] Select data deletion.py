# %%
import random
import itertools
import os
import time
import numpy as np
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

# for all scenarios run simulation
for type, config in node_types.items():

    # use common random number across scenarios
    SimRNG.ZRNG = SimRNG.InitializeRNSeed()
    rg1.seed(1)
    rg2.seed(2)

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
    env.run(until=100)

    # save results
    print('saving results ...')
    dest_path = os.getcwd()
    results_path = os.path.join(dest_path, 'Simulation Results')

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with change_dir(results_path):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        sim_dir = 'simulation_' + type + '_' + timestr
        os.mkdir(sim_dir)

        pod_generator.save_results(sim_dir)
        cluster.save_results(sim_dir)

    print(f'results for config type {type} are saved at {results_path}')

    # print results
    cluster_res = cluster.get_results()
    pod_gen_res = pod_generator.get_results()
    # avg cluster cost
    avg_cost = cluster_res['cluster' + type].cost
    success_prob = pod_gen_res['Static Pod Generator'].success_probability

    # print cost
    print(f"Cost for config type {type}={avg_cost}")
    print(f"Sucess Probability for config type {type}={success_prob}")

# %%
