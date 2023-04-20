import torch
import gymnasium as gym
import gym_STAR
from ppo_seperate import Actor
from ppo_seperate import Critic
from ppo_seperate import make_env_single
import numpy as np
from gym_normalize import NormalizeObservation

actor = torch.load("model/gym_STAR/My_Env-v1__ppo_seperate__2200__1681852772.pt")

f = open("model/gym_STAR/My_Env-v1__ppo_seperate__2200__1681852772.txt")
data = f.readlines()
f.close()

str = ' '
data = str.join(data)
data = data.replace("\n", "").replace("  ", " ")

data = np.fromstring(data, dtype=float, sep=" ")
idx = np.where(data==1e3)[0][0]
mean = data[:idx]
var = data[idx+1:]

print(mean)
print(var)

env = make_env_single("gym_STAR/My_Env-v1", "human")
env = gym.wrappers.ClipAction(env)
normalize = NormalizeObservation
env = normalize(env)
env.obs_rms.mean = mean
env.obs_rms.var = var

# actor = Actor
# model.apply(actor)

device = torch.device("cpu")

n = 3
steps = 0
env.obs_rms.mean = mean
env.obs_rms.var = var
reward = 0
for _ in range(n):
    done = False
    gamma = 1
    s, _ = env.reset(seed=2200)
    # s = (s - mean) / np.sqrt(1e-8 + var)
    while not done:
        steps += 1
        action, logprob, _ = actor.get_action(torch.unsqueeze(torch.Tensor(s), 0))
        s_, r, terminated, truncated, _ = env.step(torch.Tensor(action))
        # s_ = (s_ - mean) / np.sqrt(1e-8 + var)
        print(r)
        s = s_
        reward += r * gamma
        gamma *= 0.99
        done = np.logical_or(terminated, truncated)
    print(reward)

env.close()
