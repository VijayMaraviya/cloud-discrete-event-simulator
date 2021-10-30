import matplotlib.pyplot as plt
import csv
import os
import itertools
import logging
from collections import namedtuple

# custom classes
from SimStatistics import CTStat, DTStat
from Pod import StaticPod
from utils import NodeMatchNotFound, change_dir

# simpy classes
import simpy.core as core

# setup logger
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(message)s')

file_handler = logging.FileHandler('Simulation.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# setup plot style
plt.style.use('seaborn')


class PodGenerator:
    # value of indicator function for pod sucess/failure event
    SUCCESS = 1
    FAILURE = 0

    def __init__(self, env, name, arrival_gen):
        self._env = env
        # arrival_gen yields interarrival time when next method is called
        self.t_gen = arrival_gen
        self._name = name

        # statistics (success probability)
        self._success_metric = DTStat(env, env.now)
        self._success_prob = None

    # TODO: add properties

    @property
    def success_prob(self):
        return self._success_prob

    def generate(self, resource_guarantee,
                 resource_limit, period_gen, resource_gen,
                 schedular):

        # TODO: user should provide pod class, change it.
        # TODO: user should provide schedular object, change it
        for i in itertools.count(1):

            # create a new pod
            pod = StaticPod(self._env, 'Pod %d' % i,
                            resource_guarantee, resource_limit)

            # set the create_time
            pod.create_time = self._env.now

            # DEBUG: When a pod is created
            logger.debug(
                f"Time= {pod.create_time} | {pod.name} is created")

            # start the pod process
            process = self._env.process(
                pod.process(period_gen, resource_gen, schedular))

            # callback to report the result back to the generator
            process.callbacks.append(self._result)
            process.pod = pod

            # resume after interarrival time to submit next pod
            yield self._env.timeout(next(self.t_gen))

    def _result(self, process):
        '''
        records the pod success/failures
        '''
        if process.pod.success:
            # record value of indicator function (1 for success)
            self._success_metric.Record(PodGenerator.SUCCESS)

        else:
            # record value of indicator function (0 for failure)
            self._success_metric.Record(PodGenerator.FAILURE)

        self.record()

    def record(self):
        # calculate the results
        self._success_prob = self._success_metric.Mean()

    def clear(self):
        self._success_metric.Clear()
        self.record()

    def get_results(self):
        self.record()

        PodResult = namedtuple('PodResults', ['success_probability'])

        results = PodResult(self._success_prob)

        return {self._name: results}

    def save_results(self, destination=None):
        # record statistics
        self._success_metric.Record(0)

        if destination is None:
            destination = os.getcwd()

        # inside destination directiory
        with change_dir(destination):
            pod_gen_dir = self._name
            # create a directory for pod generator
            os.mkdir(pod_gen_dir)

            # inside the pod gen directory
            with change_dir(pod_gen_dir):

                # data (metrics)
                sim_time = self._success_metric.T_list
                indicator_vals = self._success_metric.X_list
                success_prob_r_avg = self._success_metric.RunningAvg
                n_pods = range(1, len(indicator_vals) + 1)

                row_list = zip(sim_time, indicator_vals, success_prob_r_avg)

                target_file = self._name + '-Metrics.csv'

                with open(target_file, 'w', newline='') as new_file:

                    fieldnames = ['sim_time', 'success_ind_val',
                                  'success_prob_running_avg']

                    csv_writer = csv.writer(new_file)

                    csv_writer.writerow(fieldnames)
                    for row in row_list:
                        csv_writer.writerow(row)

                # create plots directory
                os.mkdir('plots')

                # inside plots directory
                with change_dir('plots'):
                    # save plots
                    # self._plot_and_save(sim_time, indicator_vals,
                    #                     'simulation time', 'success', plots_path)
                    self._plot_and_save(sim_time, success_prob_r_avg,
                                        'simulation time', 'running success probabilty')
                    self._plot_and_save(n_pods, success_prob_r_avg,
                                        'pod count', 'running success probabilty')

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
        ax1.set_ylim(0, 1.1)
        ax1.set_title(f'{ylabel} vs {xlabel} for {self._name}')

        # save file at destination directory
        fig1.savefig(os.path.join(
            destination, xlabel + '_vs_' + ylabel + '.png'))

        # close the figure
        plt.close(fig1)
