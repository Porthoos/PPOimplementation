import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt


def smooth(data, window):
    length = len(data)
    result = np.ndarray
    for _ in range(window-1):
        data = np.append(data[0], data)
    for i in range(length):
        result = np.append(result, np.average(data[i:i+window]))
    return result[1:]



def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


def read_data(path, scalars, window):
    data = parse_tensorboard(path, scalars)
    data = data[scalars[0]].to_numpy()
    # data = data[:, 1:]
    x = data[:, 1]
    y = data[:, 2]
    y = smooth(y, window)
    return x, y

# a = ["eval/test_episode_return"]
# data = parse_tensorboard("eva/runs/gym_STAR/My_Env-v1__ppo_normalizeLayer__2200__1682584545", a)
# data = data[a[0]].to_numpy()
# # data = data[:, 1:]
# x = data[:, 1]
# y = data[:, 2]
# y = smooth(y, 100)

# x1, y1 = read_data("eva/runs/gym_STAR/My_Env-v1__ppo_normalizeLayer__2200__1682584545", ["eval/test_episode_return"], 100)
# x2, y2 = read_data("eva/runs/gym_STAR/Fix_Pos-v1__ppo_normalizeLayer__2200__1682772182", ["eval/test_episode_return"], 100)
# x3, y3 = read_data("eva/runs/gym_STAR/Fix-v1__ppo_normalizeLayer__2200__1682866328", ["eval/test_episode_return"], 100)

x1, y1 = read_data("runs/gym_STAR/My_Env-v1__ppo_normalizeLayer__2200__1682157676", ["eval/test_episode_return"], 100)
x2, y2 = read_data("runs/gym_STAR/Fix_Pos-v1__ppo_normalizeLayer__2200__1682318791", ["eval/test_episode_return"], 100)
x3, y3 = read_data("runs/gym_STAR/Fix-v1__ppo_normalizeLayer__2200__1682419347", ["eval/test_episode_return"], 100)
x4, y4 = read_data("runs/gym_STAR/RIS_Env-v1__ppo_normalizeLayer__2200__1682253681", ["eval/test_episode_return"], 100)


print("-")

plt.style.use('_mpl-gallery')

plt.figure()
plt.xlim((0, 25000000))
plt.ylim((50, 250))
x_ticks = np.linspace(0, 25000000, 11)
y_ticks = np.linspace(50, 250, 5)
plt.xticks(x_ticks, np.array(x_ticks/50, dtype=int))
plt.yticks(y_ticks)
plt.xlabel("episodes")
plt.ylabel("reward")

plt.step(x1, y1, color="blue", linewidth=0.5, label="STAR-RIS deployment")
plt.step(x2, y2, color="green", linewidth=0.5, label="STAR-RIS fix position")
plt.step(x3, y3, color="red", linewidth=0.5, label="STAR-RIS fix position and orientation")
plt.step(x4, y4, color="yellow", linewidth=0.5, label="no STAR-RIS")
plt.legend(loc="best")
plt.show()
# plt.imsave("eva/figures/data")
plt.savefig("eva/figures/data")

