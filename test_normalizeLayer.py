import torch
import gymnasium as gym
import gym_STAR
from ppo_seperate import Actor
from ppo_seperate import Critic
from ppo_normalizeLayer import make_env_single
import numpy as np

info = np.load('model/gym_STAR/My_Env-v1__ppo_normalizeLayer__2200__1682157690.npy', allow_pickle=True)
actor = torch.load("model/gym_STAR/My_Env-v1__ppo_normalizeLayer__2200__1682249979.pt")
info = info.tolist()
print(info["FD"])
env = make_env_single("gym_STAR/My_Env-v1", render_mode="human", FD=info["FD"])
env = gym.wrappers.ClipAction(env)

# actor = Actor
# model.apply(actor)

device = torch.device("cpu")

n = 3
steps = 0
reward = 0
infos = []
count = 0
for _ in range(n):
    done = False
    gamma = 1
    s, _ = env.reset(seed=2200)
    # s = (s - mean) / np.sqrt(1e-8 + var)
    while not done:
        steps += 1
        action, logprob, _ = actor.get_action(torch.unsqueeze(torch.Tensor(s), 0))
        s_, r, terminated, truncated, info = env.step(torch.Tensor(action))
        # infos[count] = info
        # count += 1
        infos.append(np.array(info["rate"]))
        count += 1
        # s_ = (s_ - mean) / np.sqrt(1e-8 + var)
        print(r)
        s = s_
        reward += r * gamma
        gamma *= 0.99
        done = np.logical_or(terminated, truncated)
    print(reward)

env.close()
np.save("infos", infos)
