import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt


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

a = ["eval/test_episode_return"]
data = parse_tensorboard("runs/gym_STAR/Fix-v1__ppo_normalizeLayer__2200__1682419347", a)
data = data[a[0]].to_numpy()
# data = data[:, 1:]
x = data[:, 1]
y = data[:, 2]

print("-")

plt.style.use('_mpl-gallery')

# plot
fig, ax = plt.subplots()

ax.step(x, y, linewidth=0.5)

# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()

