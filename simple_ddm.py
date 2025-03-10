import numpy as np

class SimpleDDM:

    def __init__(self, ):
        self.X = 0
        self.convergence_rate = 1
        self.bound = 1
        self.volatility = 1

        self.t = 1
        self.dt = 10 ** (-3)  # miliseconds

    def step(self,):
        drift_rate = self.convergence_rate * self.dt
        self.X += drift_rate + self.volatility * self.noise()
        return self.X

    def trial(self,): # Until convergence
        self.reset_paramaters()
        X = [0]
        while X[-1] < self.bound:
            self.t += 1
            X += [self.step()]
        return np.array(X)

    def noise(self, mean=0, std=1):
        return np.random.normal(mean, std) * np.sqrt(self.dt, dtype=np.float32)

    def reset_paramaters(self,):
        self.X = 0
        self.t = 1

class BUSA:

    def __init__(self, ):
        self.X = 0
        self.v_left = 0.5
        self.v_right = 0.5
        self.urgency = 1
        self.convergence_rate = 1
        self.bound_left = 1
        self.bound_right = -self.bound_left
        self.t = 1
        self.dt = 10 ** (-3)

    def step(self,):
        v_sum = self.v_left + self.v_right
        v_diff = self.v_left - self.v_right
        self.X += self.urgency * v_sum * v_diff * self.dt + self.noise()
        return self.X

    def trial(self,): # Until convergence
        self.reset_parameters()
        X = [0]
        while X[-1] < self.bound_left and X[-1] > self.bound_right:
            self.t += 1
            X += [self.step()]
        return np.array(X)

    def noise(self, mean=0, std=1):
        return np.random.normal(mean, std) * np.sqrt(self.dt, dtype=np.float32)

    def reset_parameters(self):
        self.X = 0
        self.t = 1