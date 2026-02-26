import matplotlib.pyplot as plt
import numpy as np
from ddm_models import RLDDM

np.random.seed(42)


def run_coupled_trial(model1, model2):
    model1.reset_paramaters()
    model2.reset_paramaters()
    model1.update_bounds()
    model2.update_bounds()
    X1, X2 = [0], [0]
    m1_done = False
    m2_done = False

    while not (m1_done and m2_done):
        x1, x2 = model1.X, model2.X  # snapshot both before stepping

        if not m1_done:
            model1.step(X_partner=x2)
            model1.update_bounds()
            if model1.X >= model1.upper_bound or model1.X <= model1.lower_bound:
                m1_done = True

        if not m2_done:
            model2.step(X_partner=x1)
            model2.update_bounds()
            if model2.X >= model2.upper_bound or model2.X <= model2.lower_bound:
                m2_done = True

        X1.append(model1.X)
        X2.append(model2.X)

    return np.array(X1), np.array(X2)


def run_joint_task(model1, model2, n_steps=100, time_window=4000, coupled=False):
    value_map1 = []
    value_map2 = []
    X1_series = []
    X2_series = []
    joint_rewards = np.zeros(n_steps)
    sides1 = []
    sides2 = []

    for step in range(n_steps):
        if coupled:
            X1, X2 = run_coupled_trial(model1, model2)
        else:
            X1 = model1.trial()
            X2 = model2.trial()

        X1_series.append(X1)
        X2_series.append(X2)
        sides1.append(np.sign(model1.X))
        sides2.append(np.sign(model2.X))

        if abs(model1.t - model2.t) <= time_window and np.sign(model1.X) == np.sign(
            model2.X
        ):
            joint_rewards[step] = 1

        model1.update_values(reward=joint_rewards[step])
        model2.update_values(reward=joint_rewards[step])

        value_map1.append((model1.v1, model1.v2))
        value_map2.append((model2.v1, model2.v2))

    print(f"Total JOINT rewards: {int(joint_rewards.sum())} / {n_steps}")
    return X1_series, X2_series, value_map1, value_map2, joint_rewards, sides1, sides2


def plot_evidence(ax, X_series, model):
    xlim = 20000
    xstep = 2000
    for X in X_series:
        t = np.arange(0, X.shape[0])
        ax.plot(t, X)
    ax.set_xticks(
        np.arange(0, xlim + 1, xstep), np.arange(0, xlim + 1, xstep), fontsize=14
    )
    ax.set_yticks(np.arange(-1, 1.1, 0.5), np.arange(-1, 1.1, 0.5), fontsize=14)
    ax.set_xlim(0, xlim)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Time (ms)", fontsize=18)
    ax.set_ylabel("X", fontsize=18)
    ax.axhline(1, linestyle="--", color="black")
    ax.set_title(f"{type(model).__name__}")


def plot_evidence_both(ax1, ax2, X1_series, X2_series, model1, model2):
    plot_evidence(ax1, X1_series, model1)
    plot_evidence(ax2, X2_series, model2)


def plot_side_preference(ax, value_map1, value_map2):
    vm1 = np.array(value_map1)
    vm2 = np.array(value_map2)
    ax.plot(vm1[:, 0] - vm1[:, 1], label="Model 1")
    ax.plot(vm2[:, 0] - vm2[:, 1], label="Model 2")
    ax.axhline(0, linestyle="--", color="black", linewidth=0.8)
    ax.set_xlabel("Trial", fontsize=14)
    ax.set_ylabel("v1 - v2", fontsize=14)
    ax.set_title("Side preference")
    ax.legend()


def plot_reward_moving_avg(ax, joint_rewards, window=20):
    moving_avg = np.convolve(joint_rewards, np.ones(window) / window, mode="valid")
    ax.plot(moving_avg)
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="ceiling")
    ax.set_xlabel("Trial", fontsize=14)
    ax.set_ylabel("Joint reward rate", fontsize=14)
    ax.set_title(f"Moving avg reward (window={window})")
    ax.set_ylim(0, 1)
    ax.legend()


def plot_side_agreement(ax, sides1, sides2, window=20):
    agreement = (np.array(sides1) == np.array(sides2)).astype(float)
    moving_avg = np.convolve(agreement, np.ones(window) / window, mode="valid")
    ax.plot(moving_avg)
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="chance")
    ax.set_xlabel("Trial", fontsize=14)
    ax.set_ylabel("Proportion same side", fontsize=14)
    ax.set_title(f"Side agreement (window={window})")
    ax.set_ylim(0, 1)
    ax.legend()


def plot_reward_sequences(ax, model1, model2):
    model1_rewards = np.where(np.array(model1.reward_seq))
    model2_rewards = np.where(np.array(model2.reward_seq))
    ax.scatter(model1_rewards, np.zeros_like(model1_rewards), marker="|")
    ax.scatter(model2_rewards, np.ones_like(model2_rewards), marker="|")


def plot_value_maps(value_map1, value_map2):
    value_map1 = np.array(value_map1).T
    value_map2 = np.array(value_map2).T
    fig, ax = plt.subplots(figsize=(24, 12))
    h = ax.hist2d(
        value_map1[0],
        value_map1[1],
        bins=100,
        range=[[0, 1], [0, 1]],
        density=True,
        label="Model1",
    )
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label("Density")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Value 1")
    ax.set_ylabel("Value 2")
    return fig


def plot_comparison(results_base, results_coup, window=20):
    _, _, vm1_b, vm2_b, rewards_b, sides1_b, sides2_b = results_base
    _, _, vm1_c, vm2_c, rewards_c, sides1_c, sides2_c = results_coup
    vm1_b, vm2_b = np.array(vm1_b), np.array(vm2_b)
    vm1_c, vm2_c = np.array(vm1_c), np.array(vm2_c)

    fig, ax = plt.subplots(1, 3, figsize=(36, 12))

    # Side preference: v1 - v2 per model, per condition
    ax[0].plot(vm1_b[:, 0] - vm1_b[:, 1], color="C0", alpha=0.8, label="Uncoupled M1")
    ax[0].plot(
        vm2_b[:, 0] - vm2_b[:, 1],
        color="C0",
        alpha=0.4,
        linestyle="--",
        label="Uncoupled M2",
    )
    ax[0].plot(vm1_c[:, 0] - vm1_c[:, 1], color="C1", alpha=0.8, label="Coupled M1")
    ax[0].plot(
        vm2_c[:, 0] - vm2_c[:, 1],
        color="C1",
        alpha=0.4,
        linestyle="--",
        label="Coupled M2",
    )
    ax[0].axhline(0, linestyle="--", color="black", linewidth=0.8)
    ax[0].set_xlabel("Trial", fontsize=14)
    ax[0].set_ylabel("v1 - v2", fontsize=14)
    ax[0].set_title("Side preference")
    ax[0].legend()

    # Side agreement rate
    ma_b = np.convolve(
        (np.array(sides1_b) == np.array(sides2_b)).astype(float),
        np.ones(window) / window,
        mode="valid",
    )
    ma_c = np.convolve(
        (np.array(sides1_c) == np.array(sides2_c)).astype(float),
        np.ones(window) / window,
        mode="valid",
    )
    ax[1].plot(ma_b, label="Uncoupled")
    ax[1].plot(ma_c, label="Coupled")
    ax[1].axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="chance")
    ax[1].set_xlabel("Trial", fontsize=14)
    ax[1].set_ylabel("Proportion same side", fontsize=14)
    ax[1].set_title(f"Side agreement (window={window})")
    ax[1].set_ylim(0, 1)
    ax[1].legend()

    # Joint reward moving average
    ma_r_b = np.convolve(rewards_b, np.ones(window) / window, mode="valid")
    ma_r_c = np.convolve(rewards_c, np.ones(window) / window, mode="valid")
    ax[2].plot(ma_r_b, label="Uncoupled")
    ax[2].plot(ma_r_c, label="Coupled")
    ax[2].axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="ceiling")
    ax[2].set_xlabel("Trial", fontsize=14)
    ax[2].set_ylabel("Joint reward rate", fontsize=14)
    ax[2].set_title(f"Moving avg reward (window={window})")
    ax[2].set_ylim(0, 1)
    ax[2].legend()

    fig.suptitle("Uncoupled (sequential) vs Coupled (simultaneous + κ)", fontsize=20)
    return fig


if __name__ == "__main__":
    # Uncoupled baseline: sequential trials, kappa fixed at 0
    np.random.seed(42)
    model1_base = RLDDM()
    model2_base = RLDDM()
    results_base = run_joint_task(
        model1_base, model2_base, n_steps=500, time_window=4000, coupled=False
    )

    # Coupled: simultaneous trials, kappa learned from joint reward
    np.random.seed(42)
    model1_coup = RLDDM(lr_kappa=0.1)
    model2_coup = RLDDM(lr_kappa=0.1)
    results_coup = run_joint_task(
        model1_coup, model2_coup, n_steps=500, time_window=4000, coupled=True
    )

    plot_comparison(results_base, results_coup)
    plt.show()
