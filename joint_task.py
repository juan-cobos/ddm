import time

import matplotlib.pyplot as plt
import numpy as np

from ddm_models import RLDDM

np.random.seed(42)

model1 = RLDDM()
model2 = RLDDM()

joint_rewards = 0
time_window = 4000
value_map1 = []
value_map2 = []

fig, ax = plt.subplots(1, 2, figsize=(24, 12))
steps = 100
for _ in range(steps):
    X1 = model1.trial()
    X2 = model2.trial()

    # Reward rule
    if abs(model1.t - model2.t) <= time_window and np.sign(model1.X) == np.sign(
        model2.X
    ):
        joint_rewards += 1
        model1.update_values(reward=1)
        model2.update_values(reward=1)
    else:
        model1.update_values(reward=0)
        model2.update_values(reward=0)

    value_map1.append((model1.v1, model1.v2))
    value_map2.append((model2.v1, model2.v2))
    print(
        f"Values model1: [{model1.v1:.3f}][{model1.v2:.3f}] ------ Values model2: [{model2.v1:.3f}][{model2.v2:.3f}]"
    )
    print(f"Time model1:{model1.t} ----- Time model2: {model2.t}")
    print(f"Evidence model1:{model1.X:.3f} ----- Evidence model2: {model2.X:3f}")
    t = np.arange(0, X1.shape[0])
    ax[0].plot(t, X1)

print("Total JOINT rewards", joint_rewards)

# ------- Plot Evidence model1 and joint reward sequence ----------
ypad = 0.1
xlim = 20000
xstep = 2000
ax[0].set_xticks(
    np.arange(0, xlim + 1, xstep), np.arange(0, xlim + 1, xstep), fontsize=14
)
ax[0].set_yticks(np.arange(-1, 1.1, 0.5), np.arange(-1, 1.1, 0.5), fontsize=14)
ax[0].set_xlim(0, xlim)
ax[0].set_ylim(-1.1, 1.1)
ax[0].set_xlabel("Time (ms)", fontsize=18)
ax[0].set_ylabel("X", fontsize=18)
ax[0].axhline(1, linestyle="--", color="black")
ax[0].set_title(f"{type(model1).__name__}")

model1_rewards = np.where(np.array(model1.reward_seq))
model2_rewards = np.where(np.array(model2.reward_seq))
ax[1].scatter(model1_rewards, np.zeros_like(model1_rewards), marker="|")
ax[1].scatter(model2_rewards, np.ones_like(model2_rewards), marker="|")

# ------- Plot value maps ----------

value_map1 = np.array(value_map1).T
value_map2 = np.array(value_map2).T
fig2, ax2 = plt.subplots(figsize=(24, 12))
# ax2.plot(value_map1[0], value_map1[1], "-o", label="Model 1")
# ax2.plot(value_map2[0], value_map2[1], "-o", label="Model 2")

h = ax2.hist2d(
    value_map1[0],
    value_map1[1],
    bins=100,
    range=[[0, 1], [0, 1]],
    density=True,
    label="Model1",
)
cbar = fig2.colorbar(h[3], ax=ax2)
cbar.set_label("Density")

# ax2.hist2d(value_map2[0], value_map2[1], bins=100, range=[[0, 1], [0, 1]], density=True, label="Model2")

ax2.set_ylim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_xlabel("Value 1")
ax2.set_ylabel("Value 2")
# ax2.legend()
plt.show()
