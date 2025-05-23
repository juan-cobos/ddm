"""

    Drift Difussion Model - 

    07/01/2025 - Juan Cobos

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wald

# Diffusion Decision Model (DDM)

np.random.seed(seed=42)

class DDM():
    def __init__(self, learning_rate=0.01, decay_rate=1):
        self.v_left = 0.5
        self.v_right = 0.5
        self.convergence_rate = 1
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.bias = 0
        self.dt = 10**(-3)
        self.time_range = (2*1000, 4*1000)
        self.bound_scaling = 1
        self.action_time = []
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

    def run(self, n_trials = 100, plot=False):
        trial = 0
        side_preference_lst = []

        # Reset params before each simulation
        self.reset_experiment_params()
        self.reset_trial_params()
        
        while trial < n_trials:

            drift_rate = self.convergence_rate * (self.v_left - self.v_right) * self.dt
            self.relative_value_signal += (drift_rate + self.noise() + self.bias)
            self.rvs_lst.append(self.relative_value_signal)

            # Set boundaries
            def compute_bounds():
                bound = self.bound_scaling * np.exp(-self.decay_rate * (self.v_left + self.v_right) * self.t * self.dt)
                return bound, -bound

            upper_bound, lower_bound = compute_bounds()

            if self.relative_value_signal >= upper_bound or self.relative_value_signal <= lower_bound: # If it crosses the threshold

                def update_value():
                    if  self.time_range[0] <= self.t <= self.time_range[1]:  # If within time_range
                        reward = 1
                        self.n_rewards += 1
                    else:  # Outside time_range
                        reward = 0

                    if self.relative_value_signal >= upper_bound:  # Chose left action
                        self.v_left += self.learning_rate * (reward - self.v_left)
                        side_preference_lst.append(1)

                    elif self.relative_value_signal <= lower_bound:  # Chose right action
                        self.v_right += self.learning_rate * (reward - self.v_right)
                        side_preference_lst.append(-1)

                update_value()

                # Save trial data 
                self.action_time.append(self.t)
                self.total_rvs.append(self.rvs_lst)
                self.times.append(self.t)
                # Reset trial params
                self.reset_trial_params()
                trial += 1

            self.t += 1

        #print("Rewards obtained:", self.n_rewards)
        mean_side_preference = np.sum(side_preference_lst) / len(side_preference_lst)

        if plot:

            fig, ax = plt.subplots(3, 1, figsize=(24, 12))
            #fig.tight_layout()
            
            max_rt = max(self.times)
            min_rt = min(self.times)

            # Fit Wald distribution
            loc, scale = wald.fit(self.times)    
            x = np.linspace(min_rt, max_rt, 100)
            theory = wald.pdf(x, loc, scale)

            # Plot each trial relative value signal
            for rvs in self.total_rvs:
                ax[0].plot(rvs)

            ax[0].tick_params(axis='both', which='major', labelsize=14)
            ax[0].set_xticks(np.arange(0, max_rt, round(max_rt/10)), labels=np.arange(0, max_rt, round(max_rt/10)))
            ax[0].set_xlabel("Time (ms)", fontsize=14)
            ax[0].set_ylabel("Relative Value Signal", fontsize=14)
            #ax[0].axhline(1, linestyle="--", color="black")
            #ax[0].axhline(-1, linestyle="--", color="black")

            # Compute histogram as probabilities
            hist, bin_edges = np.histogram(self.times, )
            bin_probability = hist / np.sum(hist)
            bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
            bin_width = bin_edges[1]-bin_edges[0]

            ax[1].bar(bin_middles, bin_probability, width=bin_width)
            ax[1].axvline(self.time_range[0], c="black", linestyle="--")
            ax[1].axvline(self.time_range[1], c="black", linestyle="--")
            #ax[1].hist(self.times, density=True, stacked=True) # Plot density function to show the Wald fit
            #ax[1].plot(x, theory)
            ax[1].set_ylabel("Probability", fontsize=14)
            ax[1].set_xlabel("Time (ms)", fontsize=14)
            ax[1].tick_params(axis='both', which='major', labelsize=14)
                    
            ax[2].plot(side_preference_lst)
            ax[2].set_ylabel("Side preference", fontsize=14)
            ax[2].set_xlabel("Trial", fontsize=14)
            plt.subplots_adjust(hspace=0.3)
            plt.suptitle(f"DDM two-sided",fontsize=24)
            plt.show()

#model = DDM(learning_rate=0.005)
#model.run(n_trials=500)


def fixed_values_simulation():
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    fig, ax = plt.subplots(2, 1)
    for lr in learning_rates:
        model = DDM(learning_rate=lr)
        
        v = [0.1, 0.2, 0.5, 0.7, 0.9]
        v_left = [0.1, 0.2, 0.5, 0.7, 0.9]
        v_right = [0.9, 0.7, 0.5, 0.2, 0.1]

        times = []
        n_rewards = []
        for vi in v:
            model.v_left = vi
            model.v_right = vi
            model.run(n_trials=500)
            mean_time = np.mean(model.times)
            n_rewards.append(model.n_rewards)
            times.append(mean_time)

            #print("V_left", model.v_left, "----- V_right", model.v_right)
        ax[0].plot(v, times, "o-", label=f"LR={lr}")
        ax[1].plot(v, n_rewards, "o-", label=f"LR={lr}")

        ax[0].set_ylim(0, 8500)
        ax[0].set_ylabel("Mean time (ms)", fontsize=14)
        ax[0].set_xlabel("Value", fontsize=14)
        ax[0].axhline(model.time_range[0], c="black", linestyle="--")
        ax[0].axhline(model.time_range[1], c="black", linestyle="--")
        ax[0].legend()

        ax[1].set_ylim()
        ax[1].set_ylabel("Rewards", fontsize=14)
        ax[1].set_xlabel("Value", fontsize=14)
        ax[1].legend()

        fig.suptitle("Same init values - Varying learning rate")
        #plt.savefig("Updating values.png")
        plt.show()

