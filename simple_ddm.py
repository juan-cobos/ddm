import numpy as np

class DDM:

    def __init__(self, ):
        self.X = 0
        self.convergence_rate = 1
        self.bound = 1
        self.volatility = 1

        self.t = 1
        self.dt = 10**(-3) # miliseconds

    def step(self,):
        drift_rate = self.convergence_rate * self.dt
        self.X += drift_rate + self.volatility * self.noise()
        return self.X

    def trial(self,): # Until convergence
        X = [0]
        while X[-1] < self.bound:
            self.t += 1
            X += [self.step()]
        return np.array(X)

    def noise(self, mean=0, std=1):
        return np.random.normal(mean, std) * np.sqrt(self.dt, dtype=np.float32)


