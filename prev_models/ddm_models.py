import random

import numpy as np

seed = 43
random.seed(seed)
np.random.seed(seed)


class BaseDDM:
    def noise(self, mean=0, std=0.1):
        return np.random.normal(mean, std) * np.sqrt(self.dt, dtype=np.float32)


class DriftDiffusionModel(BaseDDM):
    def __init__(self):
        self.X = 0
        self.drift_rate = 0.1
        self.bound = 1
        self.volatility = 1

        self.t = 1
        self.dt = 10 ** (-3)  # miliseconds

    def step(self):
        drift_rate = self.drift_rate * self.dt
        self.X += drift_rate + self.volatility * self.noise()
        self.t += 1
        return self.X

    def trial(self):  # Until convergence
        self.reset_paramaters()
        X = [0]
        while X[-1] < self.bound:
            X += [self.step()]
        return np.array(X)

    def reset_paramaters(self):
        self.X = 0
        self.t = 1


# TODO: Complete FullDDM and add simpleDDM, consider init -> self.t = 1 or self.t = 0
class FullDDM(BaseDDM):
    def __init__(self):
        self.X = 0
        self.v1 = 0.5
        self.v2 = 0.5
        self.drift_rate = 0.1
        self.volatility = 1
        self.bound = 1
        self.decay_rate = 0.1

        self.t = 1
        self.dt = 10 ** (-3)  # miliseconds

    def step(self):
        self.X += self.drift_rate * (self.v1 - self.v2) * self.dt + self.noise()
        self.t += 1
        return self.X

    def trial(self):  # Until convergence
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

    def reset_paramaters(self):
        self.X = 0
        self.t = 1


class RLARD(BaseDDM):  # Advantage Racing Diffusion
    def __init__(self):
        self.x1 = 0
        self.x2 = 0

        self.v0 = 1
        self.Q1 = random.random()
        self.Q2 = random.random()
        self.wd = random.random()
        self.ws = random.random()
        self.bound = 1

        self.t = 0
        self.dt = 10 ** (-3)

        self.learning_rate = 0.1

    def step(self):
        v0, Q1, Q2, wd, ws = self.v0, self.Q1, self.Q2, self.wd, self.ws
        self.x1 += (v0 + wd * (Q1 - Q2) + ws * (Q1 + Q2)) * self.dt + self.noise()
        self.x2 += (v0 + wd * (Q2 - Q1) + ws * (Q1 + Q2)) * self.dt + self.noise()
        self.t += 1

    def trial(self):  # Until convergence
        self.reset_paramaters()
        x1 = [0]
        x2 = [0]
        while self.x1 < self.bound and self.x2 < self.bound:
            self.step()
            x1 += [self.x1]
            x2 += [self.x2]
        return np.array(x1), np.array(x2)

    def reset_paramaters(self):
        self.x1 = 0
        self.x2 = 0
        self.t = 1

    def update_values(self, reward=1):

        assert self.x1 >= self.bound or self.x2 >= self.bound, (
            "None of the accumulators has reached"
        )

        if self.x1 >= self.x2:
            self.x1 += self.learning_rate * (reward - self.x1)
        else:
            self.x2 += self.learning_rate * (reward - self.x2)


class RLDDM(BaseDDM):
    def __init__(self, lr_kappa=0.0):
        self.X = 0
        self.action_time = None
        self.action_side = None  # [-1, 1]
        self.v1 = 0.5  # random.random()
        self.v2 = 0.5  # random.random()
        self.drift_rate = 0.1

        self.t = 1
        self.dt = 10 ** (-3)  # miliseconds

        self.decay_rate = 0.1
        self.bound = 1
        self.upper_bound = self.bound * np.exp(
            -self.decay_rate * (self.v1 + self.v2) * self.t * self.dt
        )
        self.lower_bound = -self.upper_bound

        self.rewards = 0
        self.reward_seq = []
        self.learning_rate = 0.1
        self.kappa = 0.0
        self.lr_kappa = lr_kappa

    def update_bounds(self):
        self.upper_bound = self.bound * np.exp(
            -self.decay_rate * (self.v1 + self.v2) * self.t * self.dt
        )
        self.lower_bound = -self.upper_bound

    def step(self, X_partner=0):
        coupling = self.kappa * (X_partner - self.X) * self.dt
        self.X += (
            self.drift_rate * (self.v1 - self.v2) * self.dt + coupling + self.noise()
        )
        self.t += 1
        return self.X

    def trial(self):  # Until convergence
        self.reset_paramaters()
        X = [0]
        while self.upper_bound > X[-1] > self.lower_bound:
            X += [self.step()]
            self.update_bounds()
        return np.array(X)

    def update_values(self, reward):
        self.rewards += reward
        if self.X >= self.upper_bound:
            self.v1 += self.learning_rate * (reward - self.v1)
        elif self.X <= self.lower_bound:
            self.v2 += self.learning_rate * (reward - self.v2)
        self.kappa += self.lr_kappa * (reward - self.kappa)
        self.reward_seq.append(reward)

    def reset_paramaters(self):
        self.X = 0
        self.t = 1


class MetaRLDDM(BaseDDM):
    def __init__(
        self,
    ):
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

    def step(self):
        self.X += self.drift_rate * (self.v1 - self.v2) * self.dt + self.noise()
        self.t += 1
        return self.X

    def trial(self):  # Until convergence
        self.reset_paramaters()
        X = [0]

        def compute_bounds():
            bound = self.bound * np.exp(
                -self.decay_rate * (self.v1 + self.v2) * self.t * self.dt
            )  # TODO: Overflow error
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

        # update_values()
        return np.array(X)

    def noise(self, mean=0, std=0.1):
        noise = np.random.normal(mean, std) * np.sqrt(self.dt, dtype=np.float32)
        self.cumulative_noise += abs(noise)
        return noise

    def reset_paramaters(self):
        self.X = 0
        self.t = 1


class BUSA(BaseDDM):
    def __init__(self):
        self.X = 0
        self.v1 = 0.5
        self.v2 = 0.5
        self.urgency = 10
        self.bound_left = 1
        self.bound_right = -self.bound_left
        self.t = 1
        self.dt = 10 ** (-3)

    def step(self):
        self.X += (
            self.urgency * (self.v1 + self.v2) * (self.v1 - self.v2) * self.dt
            + self.noise()
        )
        self.t += 1
        return self.X

    def trial(self):  # Until convergence
        self.reset_parameters()
        X = [0]
        while X[-1] < self.bound_left and X[-1] > self.bound_right:
            X += [self.step()]
        return np.array(X)

    def reset_parameters(self):
        self.X = 0
        self.t = 1
