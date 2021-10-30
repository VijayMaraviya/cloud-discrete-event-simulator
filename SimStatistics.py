"""
Converted from VBASim Basic Classes
initially by Yujing Lin for Python 2.7
Update to Python 3 by Linda Pei & Barry L Nelson

Modified by Vijaykumar Maraviya for use with Simpy
Last update 3/25/2021
"""
import math


class CTStat():
    # Generic continuous-time statistics class
    # Note that CTStat should be called AFTER the value of the variable changes

    def __init__(self, env, Tinit, Xinit):
        # Excecuted when the CTStat object is created to initialize variables
        self.Area = 0.0
        self.Tlast = Tinit  # usually entity start time
        self.TClear = Tinit
        self.Xlast = Xinit
        self.env = env  # env.now returns the current simulation time

        # for plotting
        self.T_list = [Tinit]
        self.X_list = [Xinit]
        self.A_list = [0]
        self.t_avg_list = [0]

    def Record(self, X):
        # Update the CTStat from the last time change and keep track of previous value
        self.Area = self.Area + self.Xlast * (self.env.now - self.Tlast)
        self.Tlast = self.env.now
        self.Xlast = X

        if self.Tlast == self.T_list[-1]:
            # update X in place
            self.X_list[-1] = self.Xlast
        else:
            # add new entry to the lists
            self.T_list.append(self.Tlast)
            self.X_list.append(self.Xlast)
            self.A_list.append(self.Area)
            self.t_avg_list.append(self.Area/self.Tlast)

    def Mean(self):
        # Return the sample mean up through the current time but do not update
        mean = 0.0
        if (self.env.now - self.TClear) > 0.0:
            mean = (self.Area + self.Xlast * (self.env.now -
                                              self.Tlast)) / (self.env.now - self.TClear)
        return mean

    def Max(self):
        max_val = 0.0
        if (self.env.now - self.TClear) > 0.0:
            max_val = max(self.X_list)
        return max_val

    def Min(self):
        min_val = 0.0
        if (self.env.now - self.TClear) > 0.0:
            min_val = min(self.X_list)
        return min_val

    def Clear(self):
        # Clear statistics during the simulation
        self.Area = 0.0
        self.Tlast = self.env.now
        self.TClear = self.env.now

        self.T_list = []
        self.X_list = []
        self.A_list = []
        self.t_avg_list = []


class DTStat():
    # Generic discrete-time statistics class

    def __init__(self, env, Tinit):
        # Excecutes when the DTStat object is created to initialize variables
        self.Sum = 0.0
        self.SumSquared = 0.0
        self.NumberOfObservations = 0.0

        # to keep track of time (not use in computation)
        self.Tlast = Tinit
        self.env = env  # env.now returns the current simulation time

        # timestamp of Record method calls
        self.T_list = []
        # store indicator values
        self.X_list = []
        # store running average
        self.RunningAvg = []

    def Record(self, X):
        # Update the DTStat
        self.Sum = self.Sum + X
        self.SumSquared = self.SumSquared + X * X
        self.NumberOfObservations = self.NumberOfObservations + 1

        # time of `Record` call
        self.Tlast = self.env.now

        self.X_list.append(X)
        self.T_list.append(self.Tlast)
        self.RunningAvg.append(self.Mean())

    def Mean(self):
        # Return the sample mean
        mean = 0.0
        if self.NumberOfObservations > 0.0:
            mean = self.Sum / self.NumberOfObservations
        return mean

    def StdDev(self):
        # Return the sample standard deviation
        stddev = 0.0
        if self.NumberOfObservations > 1.0:
            stddev = math.sqrt((self.SumSquared - self.Sum**2 /
                                self.NumberOfObservations) / (self.NumberOfObservations - 1))
        return stddev

    def N(self):
        # Return the number of observations collected
        return self.NumberOfObservations

    def Clear(self):
        # Clear statistics
        self.Sum = 0.0
        self.SumSquared = 0.0
        self.NumberOfObservations = 0.0

        self.Tlast = self.env.now
        # timestamp of Record method calls
        self.T_list = []
        # store indicator values
        self.X_list = []
        # store running average
        self.RunningAvg = []
