import numpy as np

class DriftDiffusionModel:

    def __init__(self, ):
        self.X = 0
        self.drift_rate = 0.1
        self.bound = 1
        self.volatility = 1

        self.t = 1
        self.dt = 10 ** (-3)  # miliseconds

    def step(self,):
        drift_rate = self.drift_rate * self.dt
        self.X += drift_rate + self.volatility * self.noise()
        self.t += 1
        return self.X

    def trial(self,): # Until convergence
        self.reset_paramaters()
        X = [0]
        while X[-1] < self.bound:
            X += [self.step()]
        return np.array(X)

    def noise(self, mean=0, std=0.1):
        return np.random.normal(mean, std) * np.sqrt(self.dt, dtype=np.float32)

    def reset_paramaters(self,):
        self.X = 0
        self.t = 1

# TODO: Complete FullDDM and add simpleDDM, consider init -> self.t = 1 or self.t = 0
class FullDDM:
    
    def __init__(self,):
        self.X = 0
        self.v1 = 0.5
        self.v2 = 0.5
        self.drift_rate = 0.1
        self.volatility = 1
        self.bound = 1
        self.decay_rate = 0.1

        self.t = 1
        self.dt = 10 ** (-3)  # miliseconds

    def step(self, ):
        self.X += self.drift_rate * (self.v1 - self.v2) * self.dt + self.noise()
        self.t += 1
        return self.X

    def trial(self,): # Until convergence
        self.reset_paramaters()

        def compute_bounds():
            bound = self.bound * np.exp(-self.decay_rate * self.t * self.dt)
            return bound, -bound
        upper_bound, lower_bound = compute_bounds()

        X = [0]
        while X[-1] < upper_bound and X[-1] > lower_bound:
            X += [self.step()]
            upper_bound, lower_bound = compute_bounds()
        return np.array(X)

    def noise(self, mean=0, std=0.1):
        return np.random.normal(mean, std) * np.sqrt(self.dt, dtype=np.float32)

    def reset_paramaters(self, ):
        self.X = 0
        self.t = 1

class RLDDM:

    def __init__(self,):
        self.X = 0
        self.v1 = 0.5
        self.v2 = 0.5
        self.drift_rate = 0.1

        self.t = 1
        self.dt = 10 ** (-3)  # miliseconds

        self.decay_rate = 0.1
        self.bound = 1

        self.learning_rate = 0.01

    def step(self, ):
        self.X += self.drift_rate * (self.v1 - self.v2) * self.dt + self.noise()
        self.t += 1
        return self.X

    def trial(self,): # Until convergence
        self.reset_paramaters()
        X = [0]
        def compute_bounds():
            bound = self.bound * np.exp(-self.decay_rate * (self.v1 + self.v2) * self.t * self.dt)
            return bound, -bound

        upper_bound, lower_bound = compute_bounds()
        while upper_bound > X[-1] > lower_bound:
            X += [self.step()]
            upper_bound, lower_bound = compute_bounds()

        def update_values():
            # if self.t within interval reward = 1 else reward = 0
            reward = 0  # Modify if needed based on reward criteria
            if X[-1] >= upper_bound:
                self.v1 += self.learning_rate * (reward - self.v1)
            else:
                self.v2 += self.learning_rate * (reward - self.v2)
        #update_values()
        return np.array(X)

    def noise(self, mean=0, std=0.1):
        return np.random.normal(mean, std) * np.sqrt(self.dt, dtype=np.float32)

    def reset_paramaters(self, ):
        self.X = 0
        self.t = 1

class MetaRLDDM:

    def __init__(self,):
        self.X = 0
        self.v1 = 0.5
        self.v2 = 0.5
        self.drift_rate = 1
        self.volatility = 1

        self.t = 1
        self.dt = 10 ** (-3)  # miliseconds

        self.decay_rate = 0.1
        self.bound = 1

        self.learning_rate = 0.1
        self.cumulative_noise = 0

    def step(self, ):
        self.X += self.drift_rate * (self.v1 - self.v2) * self.dt + self.noise()
        self.t += 1
        return self.X

    def trial(self,): # Until convergence
        self.reset_paramaters()
        X = [0]
        def compute_bounds():
            bound = self.bound * np.exp(-self.decay_rate * (self.v1 + self.v2) * self.t * self.dt) # TODO: Overflow error
            return bound, -bound

        upper_bound, lower_bound = compute_bounds()
        while upper_bound > X[-1] > lower_bound:
            X += [self.step()]
            upper_bound, lower_bound = compute_bounds()

        def update_values():
            # if self.t within interval reward = 1 else reward = 0
            reward = 0  # TODO: Modify if needed based on reward criteria
            alpha = self.learning_rate * self.cumulative_noise
            if X[-1] >= upper_bound:
                self.v1 += alpha * (reward - self.v1)
            else:
                self.v2 += alpha * (reward - self.v2)
        #update_values()
        return np.array(X)

    def noise(self, mean=0, std=0.1):
        noise = np.random.normal(mean, std) * np.sqrt(self.dt, dtype=np.float32)
        self.cumulative_noise += abs(noise)
        return noise

    def reset_paramaters(self, ):
        self.X = 0
        self.t = 1


class BUSA:

    def __init__(self, ):
        self.X = 0
        self.v1 = 0.5
        self.v2 = 0.5
        self.urgency = 10
        self.bound_left = 1
        self.bound_right = -self.bound_left
        self.t = 1
        self.dt = 10 ** (-3)

    def step(self,):
        self.X += self.urgency * (self.v1 + self.v2) * (self.v1 - self.v2) * self.dt + self.noise()
        self.t += 1
        return self.X

    def trial(self,): # Until convergence
        self.reset_parameters()
        X = [0]
        while X[-1] < self.bound_left and X[-1] > self.bound_right:
            X += [self.step()]
        return np.array(X)

    def noise(self, mean=0, std=0.1):
        return np.random.normal(mean, std) * np.sqrt(self.dt, dtype=np.float32)

    def reset_parameters(self):
        self.X = 0
        self.t = 1