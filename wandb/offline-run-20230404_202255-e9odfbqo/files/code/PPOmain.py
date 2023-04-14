import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Beta, Normal
import torch.nn.functional as F
import wandb
from PPOimp import PPO_net
from PPOimp import replay_buffer
import gym_STAR


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--wandb-project-name", type=str, default="PPO_implementation")
    parser.add_argument("--wandb-entity", type=str, default="Aramiis")

    parser.add_argument("--policy_dist", type=str, default="Beta")

    parser.add_argument("--env-id", type=str, default="gym_STAR/My_Env-v1")
    parser.add_argument("--max_train_steps", type=int, default=3e6)
    parser.add_argument("--evaluate-freq", type=int, default=5e3)
    parser.add_argument("--lr_a", type=float, default=3e-4)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--valuef_coef", type=float, default=0.5)

    parser.add_argument("--batch_adv_norm", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--mbatch_adv_norm", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--vloss_clip",type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--use_state_norm", type=lambda x: bool(strtobool(x)), default=True)

    parser.add_argument("--hidden_width", type=int, default=64)

    args = parser.parse_args()
    return args

def make_env(args):
    env = gym.make(args.env_id)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])

    return env

def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    steps = 0
    for _ in range(times):
        s, _ = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        terminated, truncated = False, False
        episode_reward = 0
        while not (terminated or terminated):
            steps += 1
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            # if args.policy_dist == "Beta":
            #     action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            # else:
            #     action = a
            action = a
            s_, r, truncated, terminated, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times, steps / times

def main():
    args = parse_args()

    env = make_env(args)
    env_evaluate = make_env(args)
    seed = args.seed

    # env.seed(seed)
    env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    os.environ["WANDB_API_KEY"] = "1efa41085884f0f2f57e32ca6f6cd45e021f482d"
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        settings=wandb.Settings(start_method="thread"),
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        # monitor_gym=True, no longer works for gymnasium
        save_code=True,
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    state_norm = Normalization(shape=args.state_dim)
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    total_steps = 0
    evaluate_num = 0
    evaluate_reward = []

    replaybuffer = replay_buffer(args)
    agent = PPO_net(args, writer)

    while total_steps < args.max_train_steps:
        s, _ = env.reset(seed=args.seed)
        s = state_norm(s)
        reward_scaling.reset()

        episode_steps = 0
        episode_return = 0
        gamma = 1
        done = False

        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)

            action = a
            s_, r, terminated, truncated, _ = env.step(action)

            episode_return += gamma * r
            gamma *= args.gamma

            s_ = state_norm(s_)
            r = reward_scaling(r)
            done = np.logical_or(terminated, truncated)

            replaybuffer.store(s, a, a_logprob, r, s_, done, terminated)
            s = s_
            total_steps += 1

            if replaybuffer.count == args.batch_size:
                agent.update(replaybuffer, total_steps, args.state_dim, args.action_dim)
                replaybuffer.count = 0

            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, episode_length = evaluate_policy(args, env_evaluate, agent, state_norm)
                writer.add_scalar("eval/episode_return", evaluate_reward, total_steps)
                writer.add_scalar("eval/episode_length", episode_length, total_steps)
                writer.add_scalar("eval/env_mean", state_norm.running_ms.mean)
                writer.add_scalar("eval/env_std", state_norm.running_ms.std)

        if total_steps % 1e3 == 0:
            writer.add_scalar("train/episode_return", episode_return, total_steps)
            writer.add_scalar("train/episode_length", episode_steps, total_steps)

    env.close()
    env_evaluate.close()
    writer.close()





if __name__ == "__main__":
    main()