"""

    Drift Difussion Model - 

    07/01/2025 - Juan Cobos

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wald
from matplotlib.colors import Normalize
from matplotlib import cm

# Simple Diffusion Decision Model (sDDM)

np.random.seed(seed=42)

class DDM():
    def __init__(self, learning_rate=0.001, decay_rate=1):
        self.v_left = 0
        self.v_right = 1
        self.convergence_rate = 1
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.bias = 0
        self.dt = 10**(-3)
        self.time_range = (1*1000, 4*1000)
        self.bound_scaling = 1

        self.reset_experiment_params()
        self.reset_trial_params()

    def reset_experiment_params(self):
        self.times = []
        self.n_rewards = 0
        self.total_rvs = []        
    
    def reset_trial_params(self):
        self.relative_value_signal = 0
        self.rvs_lst = [0]
        self.t = 0

    def noise(self, mean=0, std=0.1):
        return np.random.normal(mean, std) * np.sqrt(self.dt)

    def update_value(self, v):
        if self.t >= self.time_range[0] and self.t <= self.time_range[1]: # If within time_range
            reward = 1
            self.n_rewards += 1
        else: # Outside time_range
            reward = 0
        #v += self.learning_rate * (reward - v)
        return v

    def run(self, n_trials = 100, plot=False):
        trial = 0
        side_preference_lst = []

        # Params must be reseted before each simulation
        self.reset_experiment_params()
        self.reset_trial_params()
        
        while trial <= n_trials:

            drift_rate = self.convergence_rate * (self.v_left - self.v_right) * self.dt
            self.relative_value_signal += (drift_rate + self.noise() + self.bias)
            self.rvs_lst.append(self.relative_value_signal)

            # Set boundaries
            bounding_decay = self.bound_scaling * np.exp(-self.decay_rate * self.t * self.dt * (self.v_left + self.v_right)) 
            upper_bound = bounding_decay
            lower_bound = -bounding_decay

            if self.relative_value_signal >= upper_bound or self.relative_value_signal <= lower_bound: # If it crosses the threshold
                    
                if self.relative_value_signal >= upper_bound : # Chose left action
                    self.v_left = self.update_value(self.v_left)
                    side_preference_lst.append(1)

                elif self.relative_value_signal <= lower_bound: # Chose right action
                    self.v_right = self.update_value(self.v_right)
                    side_preference_lst.append(-1)
        
                # Save trial data 
                self.total_rvs.append(self.rvs_lst)
                self.times.append(self.t)
                # Reset trial params
                self.reset_trial_params()
                trial += 1

            self.t += 1

        print("Rewards obtained:", self.n_rewards)
        side_preference_score = np.sum(side_preference_lst) / len(side_preference_lst)

        if plot:

            fig, ax = plt.subplots(3, 1, figsize=(24, 12))
            fig.tight_layout(pad=5.0)
            max_rt = max(self.times)
            min_rt = min(self.times)

            loc, scale = wald.fit(self.times)    
            x = np.linspace(min_rt, max_rt, 100)
            theory = wald.pdf(x, loc, scale)

            for rvs in self.total_rvs:
                ax[0].plot(rvs)

            ax[0].tick_params(axis='both', which='major', labelsize=18)
            ax[0].set_xticks(np.arange(0, max_rt, round(max_rt/10)), labels=np.arange(0, max_rt, round(max_rt/10)))
            ax[0].set_xlabel("Time (ms)", fontsize=18)
            ax[0].set_ylabel("Relative Value Signal", fontsize=18)
            #ax[0].axhline(1, linestyle="--", color="black")
            #ax[0].axhline(-1, linestyle="--", color="black")
        
            hist, bin_edges = np.histogram(self.times, )
            bin_probability = hist / np.sum(hist)
            bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
            # Compute the bin-width
            bin_width = bin_edges[1]-bin_edges[0]
            # Plot the histogram as a bar plot
            ax[1].bar(bin_middles, bin_probability, width=bin_width)
            ax[1].axvline(self.time_range[0], c="black", linestyle="--")
            ax[1].axvline(self.time_range[1], c="black", linestyle="--")
            #ax[1].hist(self.times, density=True, stacked=True) # Plot density function to show the Wald fit
            #ax[1].plot(x, theory)
            ax[1].set_ylabel("Probability", fontsize=18)
            ax[1].set_xlabel("Time (ms)", fontsize=18)
            ax[1].tick_params(axis='both', which='major', labelsize=18)
                    
            ax[2].plot(side_preference_lst)

            plt.suptitle(f"sDDM LR={self.learning_rate} DR={self.decay_rate}",fontsize=24)
            plt.show()

model = DDM()
#model.run(plot=True)


v = [0.1, 0.2, 0.5, 0.7, 0.9]
v_left = [0.1, 0.2, 0.5, 0.7, 0.9]
v_right = [0.9, 0.7, 0.5, 0.2, 0.1]

times = []
n_rewards = []
for vi in v:
    model.v_left = vi
    model.v_right = vi
    model.run()
    mean_time = np.mean(model.times)
    print(mean_time)
    n_rewards.append(model.n_rewards)
    times.append(mean_time)

fig, ax = plt.subplots(2, 1)
ax[0].scatter(v, times)
ax[0].set_ylabel("Mean time (ms)",)
ax[0].set_xlabel("Value",)
ax[0].axhline(model.time_range[0], c="black", linestyle="--")
ax[0].axhline(model.time_range[1], c="black", linestyle="--")

ax[1].scatter(v, n_rewards)
ax[1].set_ylabel("Rewards",)
ax[1].set_xlabel("Value",)

plt.savefig("Fixed values.png")
plt.show()

"""
learning_rates = [0.1, 0.05, 0.005, 0.001]
decay_rates = [0.1, 0.05, 0.005, 0.001]
mean_lst = []
std_lst = []

x = []
y = []

for lr in learning_rates:
    for dr in decay_rates:
        print("entra")
        model = sDDM(learning_rate=lr, decay_rate=dr)
        model.run()
        mean_lst.append(np.mean(model.times))
        std_lst.append(np.std(model.times))
        x.append(lr)
        y.append(dr)

max_mean = np.max(mean_lst)
max_std = np.max(std_lst)


fig = plt.figure()
ax = plt.axes(projection='3d')

# Normalize the values
norm = Normalize(vmin=0, vmax=max_mean)  # Define normalization
normalized_values = norm(mean_lst) # Map values to [0, 1]

# Use colormap to get colors
cmap = plt.cm.viridis
colors = cmap(normalized_values)

# Create the scatter plot
plot = ax.scatter(x, y, normalized_values, c=colors, s=100)  # Use normalized colors

fig.colorbar(plot)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel("Learning rate", fontsize=18)
ax.set_ylabel("Decay rate", fontsize=18)
plt.show()
"""


"""
# INIT PARAMS
relative_value_signal = 0 
v_left, v_right = 0, 0
convergence_rate = 1  
bias = 0
reward = 0.1
learning_rate = 0.01
decay_rate = 0.1
mean = 0
std = 0.1

upper_bound = 1
lower_bound = -1

trial = 0
n_trials = 100
rvs_lst = [0]
times = []
t = 0
dt = 10**(-3)

def noise():
    return np.random.normal(mean, std) * np.sqrt(dt)

side_preference_lst = []

fig, ax = plt.subplots(3, 1, figsize=(24, 12))
fig.tight_layout(pad=5.0)

while trial <= n_trials:

    t += 1

    drift_rate = convergence_rate * (v_left - v_right) * dt
    relative_value_signal += (drift_rate + noise() + bias)
    rvs_lst.append(relative_value_signal)

    bounding_decay = np.exp(-decay_rate * t) * dt
    upper_bound -= bounding_decay
    lower_bound += bounding_decay

    if relative_value_signal >= upper_bound or relative_value_signal <= lower_bound: # If it crosses the threshold
        
        if relative_value_signal >= upper_bound: # Chose left action
            v_left += learning_rate * (reward - v_left)        
            side_preference_lst.append(1)

        else: # relative_value_signal <= lower_bound: # Chose right action
            v_right += learning_rate * (reward - v_right)        
            side_preference_lst.append(-1)
        
        # Save trial data 
        trial_time = len(rvs_lst)
        times.append(trial_time)
        ax[0].plot(rvs_lst, label= "Relative Value Signal")

        # Reset trial params
        rvs_lst = [0]
        trial += 1
        relative_value_signal = 0
        upper_bound = 1
        lower_bound = -1
        t = 0

side_preference_score = np.sum(side_preference_lst) / len(side_preference_lst)

max_rt = max(times)
min_rt = min(times)

loc, scale = wald.fit(times)    
x = np.linspace(min_rt, max_rt, 100)
theory = wald.pdf(x, loc, scale)

ax[0].tick_params(axis='both', which='major', labelsize=18)
ax[0].set_xticks(np.arange(0, max_rt, round(max_rt/10)), labels=np.arange(0, max_rt, round(max_rt/10)))
ax[0].set_xlabel("Time (ms)", fontsize=18)
ax[0].set_ylabel("Relative Value Signal", fontsize=18)
ax[0].axhline(1, linestyle="--", color="black")
ax[0].axhline(-1, linestyle="--", color="black")

ax[1].hist(times, density=True, stacked=True) # Plot density function to show the Wald fit
ax[1].plot(x, theory)
ax[1].set_xlabel("Time (ms)", fontsize=18)
ax[1].tick_params(axis='both', which='major', labelsize=18)

ax[2].plot(side_preference_lst)

# Normalize to probabilities

hist, bin_edges = np.histogram(times, )
bin_probability = hist / np.sum(hist)
bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
# Compute the bin-width
bin_width = bin_edges[1]-bin_edges[0]
# Plot the histogram as a bar plot
ax[1].bar(bin_middles, bin_probability, width=bin_width)


plt.suptitle("sDDM",fontsize=24)
plt.show()
"""

