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
import gym_STAR


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = nn.Tanh()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.alpha_layer, gain=0.01)
        orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = nn.Tanh()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = nn.Tanh()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class replay_buffer():
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.v = np.zeros((args.batch_size, args.state_dim))
        self.done = np.zeros((args.batch_size, 1))
        self.done_ = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, v, done, done_):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.v[self.count] = v
        self.done[self.count] = done
        self.done_[self.count] = done_
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        v = torch.tensor(self.v, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        done_ = torch.tensor(self.done_, dtype=torch.float)

        return s, a, a_logprob, r, s_, v, done, done_


class PPO_net():
    def __int__(self, args, writer):
        self.writer = writer

        self.policy_dist = args.policy_dist

        self.batch_size = args.batch_size
        self.minibatch_size = args.minibatch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.update_epochs = args.update_epochs  # update for K epochs
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.valuef_coef = args.valuef_coef

        self.batch_adv_norm = args.batch_adv_norm
        self.mbatch_adv_norm = args.mbatch_adv_norm
        self.vloss_clip = args.vloss_clip

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().numpy().flatten()
        else:
            a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -10, 10)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, replaybuffer, total_steps, observation_space, action_space):
        s, a, a_logprob, r, s_, v, done, done_ = replaybuffer.numpy_to_tensor()
        advantages = torch.zeros_like(r)
        gae = 0

        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)
            # TODO specially for this env, there is no need to judge whether the game is truncated or terminated
            for t in reversed(range(self.batch_size)):
                # TODO done_ is specially for whether terminated or truncated. as for my env, done_ should be
                #  terminated, which is always false
                delta = r[t] + self.gamma * vs_[t] * (1 - done_[t]) - vs[t]
                # TODO whether here should be done or done_? cleanrl use done_, another use done
                gae = delta + self.gamma * self.lamda * gae * (1.0 - done[t])
                advantages[t] = gae

            returns = advantages + v
            if self.batch_adv_norm:
                advantages = (advantages - advantages.mean()) / (1e-8 + advantages.std(self))

            batch_s = s.reshape((-1,) + observation_space)
            batch_a = a.reshape((-1,) + action_space)
            batch_logprob = a_logprob.reshape(-1)
            batch_adv = advantages.reshape(-1)
            batch_return = returns.reshape(-1)
            batch_v = v.reshape(-1)

            batch_idx = np.arange(self.batch_size)
            clipfracs = []

        for _ in range(self.update_epochs):
            np.random.shuffle(batch_idx)
            v_loss, policy_loss, entropy_loss = 0, 0, 0
            for start in range(0, self.batch_size, self.minibatch_size):
                minibatch_idx = batch_idx[start:start+self.minibatch_size]
                dist = self.actor.get_dist(batch_s[minibatch_idx])
                newlogprob = dist.log_prob(batch_a[minibatch_idx]).sum(1)
                entropy = dist.entropy().sum(1)
                newvalue = self.critic(batch_s[minibatch_idx])
                logratio = newlogprob - batch_logprob[minibatch_idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.epsilon).float().mean().irem()]

                minibatch_adv = batch_adv[minibatch_idx]
                if self.mbatch_adv_norm:
                    minibatch_adv = (minibatch_adv - minibatch_adv.mean()) / (minibatch_adv.std() + 1e-8)

                policy_loss1 = -minibatch_adv * ratio
                policy_loss2 = -minibatch_adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                newvalue = newvalue.view(-1)
                if self.vloss_clip:
                    vloss_unclip = (newvalue - batch_v[minibatch_idx]) ** 2
                    v_clip = batch_v[minibatch_idx] + torch.clamp(newvalue - batch_v[minibatch_idx], -self.epsilon, self.epsilon)
                    vloss_clip = (v_clip - batch_v[minibatch_idx]) ** 2
                    vloss_max = torch.max(vloss_unclip, vloss_clip)
                    v_loss = 0.5 * vloss_max
                else:
                    v_loss = ((newvalue - batch_v[minibatch_idx]) ** 2)

                entropy_loss = entropy.mean()
                # TODO use separate optimizer / one optimizer
                actor_loss = policy_loss - self.entropy_coef * entropy_loss
                critic_loss = v_loss

                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.mean().backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

            self.lr_decay(total_steps)

            self.writer.add_scalar("charts/actor_learning_rate", self.optimizer_actor.param_groups[0]['lr'], total_steps)
            self.writer.add_scalar("charts/critic_learning_rate", self.optimizer_critic.param_groups[0]['lr'], total_steps)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), total_steps)
            self.writer.add_scalar("losses/policy_loss", policy_loss.item(), total_steps)
            self.writer.add_scalar("losses/entropy_loss", entropy_loss.item(), total_steps)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), total_steps)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), total_steps)
            self.writer.add_scalar("losses/clip_fraction", np.mean(clipfracs), total_steps)















