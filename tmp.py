import numpy as np
info = np.load('model/gym_STAR/My_Env-v1__ppo_normalizeLayer__2200__1682157690.npy', allow_pickle=True)
info = info.tolist()
data = info["FD"]
print(info)