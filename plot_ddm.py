import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from simple_ddm import *
import time

np.random.seed(42)

model = SimpleDDM()
X = model.trial()

fig, ax = plt.subplots(1, figsize=(24, 12))
t = np.arange(0, X.shape[0])

ax.plot(t, X)
animated_plot, = ax.plot([], [])

def init_plot():
    ypad = 0.1
    ax.set_xticks(np.arange(0, model.t, 100), np.arange(0, model.t, 100), fontsize=14)
    ax.set_yticks(np.arange(-1, model.X, 0.25), np.arange(-1, model.X, 0.25), fontsize=14)
    ax.set_xlim(0, max(t))
    ax.set_ylim(min(X) - ypad, max(X) + ypad)
    ax.set_xlabel("Time (ms)", fontsize=18)
    ax.set_ylabel("X", fontsize=18)
    ax.axhline(model.bound, linestyle='--', color="black")
    return animated_plot,

def update(frame):
    animated_plot.set_data(t[:frame], X[:frame])
    return animated_plot,


animation = FuncAnimation(
    fig = fig,
    func = update,
    frames = range(0, X.shape[0]+3, 3),
    init_func=init_plot,
    interval = 1,
    repeat = False,
    blit = True
)

animation.save("simple_ddm.mp4", fps=60, writer="ffmpeg")
#plt.show()
