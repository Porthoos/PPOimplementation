import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt

data = np.load("infos.npy", allow_pickle=True)

user0 = data[:50, 0]
user1 = data[:50, 1]
user2 = data[:50, 2]
user3 = data[:50, 3]
user4 = data[:50, 4]
user5 = data[:50, 5]

x = np.linspace(0, 50)

plt.style.use('_mpl-gallery')
plt.figure()

plt.xlabel("time slot / s")
plt.ylabel("communication rate / Mb")
plt.xlim((0, 50))

# plt.plot(x, user0, color="blue", linewidth=0.5, label="user1")
# plt.plot(x, user1, color="green", linewidth=0.5, label="user2")
# plt.plot(x, user2, color="red", linewidth=0.5, label="user3")
# plt.plot(x, user3, color="yellow", linewidth=0.5, label="user4")
# plt.plot(x, user4, color="orange", linewidth=0.5, label="user5")
# plt.plot(x, user5, color="purple", linewidth=0.5, label="user6")
# plt.legend(loc="best")

y = np.vstack([user1, user2, user3, user4, user5, user0])
plt.stackplot(x, user1, user2, user3, user4, user5, user0, colors=['red', 'blue', 'green', 'yellow', 'purple', 'orange'],
              labels=['user1', 'user2', 'user3', 'user4', 'user5', 'user6'], alpha=0.65)
plt.legend(loc='best')
plt.show()
