# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import gym_STAR

from gym_normalize import NormalizeObservation


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=2200,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="adoos",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="gym_STAR/Fix_Pos-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=25000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=8192,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--eval_freq", type=float, default=5e3,
        help="evaluate frequency")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def make_env_single(env_id, render_mode=None, FD=None):
    env = gym.make(env_id, render_mode=render_mode, FD=FD)
    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def evaluata_policy_train(env, agent, normalize, gamma_, seed):
    n = 3
    steps = 0
    reward = 0
    for _ in range(n):
        done = False
        gamma = 1
        s, _ = env.reset(seed=seed)
        while not done:
            steps += 1
            action, logprob, _ = agent.get_action(torch.unsqueeze(torch.Tensor(s), 0))
            s_, r, terminated, truncated, _ = env.step(torch.Tensor(action))
            s = s_
            reward += r * gamma
            gamma *= gamma_
            done = np.logical_or(terminated, truncated)
    return reward / n, steps / n


def evaluata_policy_test(env, agent, normalize, gamma_, seed):
    n = 3
    steps = 0
    reward = 0
    for _ in range(n):
        done = False
        gamma = 1
        s, _ = env.reset(seed=seed)
        while not done:
            steps += 1
            action, logprob, _ = agent.get_action(torch.unsqueeze(torch.Tensor(s), 0))
            s_, r, terminated, truncated, _ = env.step(torch.Tensor(action))
            s = s_
            reward += r * gamma
            gamma *= gamma_
            done = np.logical_or(terminated, truncated)
    return reward / n, steps / n



class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.LayerNorm(np.array(envs.observation_space.shape).prod()),
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x).double()
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


class Critic(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        os.environ["WANDB_API_KEY"] = "1efa41085884f0f2f57e32ca6f6cd45e021f482d"
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    # )
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    envs = make_env_single(args.env_id)
    envs = gym.wrappers.ClipAction(envs)
    # normalize_train = NormalizeObservation
    # envs = normalize_train(envs)
    _, info = envs.reset()

    # with open(f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}.txt", "w") as f:
    #     f.write(np.array2string(np.array(info["FD"]), formatter={'float_kind':lambda x: "%.2f" % x}))
    # np.savetxt(f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}BK.txt",
    #            np.array(info["FD"][0]).view(float))
    # np.savetxt(f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}RK.txt",
    #            np.array(info["FD"][1]).view(float))
    # np.savetxt(f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}BR.txt",
    #            np.array(info["FD"][2]).view(float))
    # print(info["FD"][0])
    # np.save(f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}BK.npy",
    #         np.array(info["FD"][0], dtype='c').view(float))
    # np.save(f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}RK.npy",
    #         np.array(info["FD"][1], dtype='c').view(float))
    # np.save(f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}BR.npy",
    #         np.array(info["FD"][2], dtype='c').view(float))
    # print(np.load(f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}BK.npy"))

    np.save(f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}.npy", info["FD"])

    evaluate_env = make_env_single(args.env_id, FD=info["FD"])
    evaluate_env = gym.wrappers.ClipAction(evaluate_env)
    # normalize_test = NormalizeObservation
    # evaluate_env = normalize_test(evaluate_env)

    actor = Actor(envs).to(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)

    critic = Critic(envs).to(device)
    optimizer_critic = optim.Adam(critic.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()

    episode_reward = 0
    episode_length = 0
    gamma = 1

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            # optimizer_critic.param_groups[0]["lr"] = lrnow
            optimizer_actor.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _ = actor.get_action(torch.unsqueeze(next_obs, 0))
                value = critic.get_value(torch.unsqueeze(next_obs, 0))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.tensor(done, dtype=float).to(device)

            if global_step % args.eval_freq == 0:
                eval_return_test, eval_length_test = evaluata_policy_test(evaluate_env, actor, envs, args.gamma, args.seed)
                eval_return_train, eval_length_train = evaluata_policy_train(envs, actor, envs, args.gamma, args.seed)
                writer.add_scalar("eval/test_episode_return", eval_return_test, global_step)
                writer.add_scalar("eval/test_episode_length", eval_length_test, global_step)
                writer.add_scalar("eval/train_episode_return", eval_return_train, global_step)
                writer.add_scalar("eval/train_episode_length", eval_length_train, global_step)
                print(f"global_step={global_step}, episodic_return={eval_return_test}")
                print(f"global_step={global_step}, episodic_return={eval_return_train}")

            # Only print when at least 1 env is done
            # if "final_info" not in infos:
            #     continue

            # for info in infos["final_info"]:
            #     # Skip the envs that are not done
            #     if info is None:
            #         continue
            #     # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #     writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #     writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            episode_length += 1
            episode_reward += gamma * reward
            gamma *= args.gamma

            if done:
                writer.add_scalar("train/episodic_return", episode_reward, global_step)
                writer.add_scalar("train/episodic_length", episode_length, global_step)
                next_obs, _ = envs.reset(seed=args.seed)
                done = False
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.tensor(done, dtype=float).to(device)
                episode_reward = 0
                episode_length = 0
                gamma = 1

        # bootstrap value if not done
        with torch.no_grad():
            next_value = critic.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = actor.get_action(b_obs[mb_inds], b_actions[mb_inds])
                newvalue = critic.get_value(b_obs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss_actor = pg_loss - args.ent_coef * entropy_loss
                loss_critic = v_loss

                optimizer_actor.zero_grad()
                loss_actor.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                optimizer_actor.step()

                optimizer_critic.zero_grad()
                loss_critic.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                optimizer_critic.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer_critic.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track and args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                    video_filenames.add(filename)

    torch.save(actor, f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}.pt")
    print(info["FD"])
    # with open(f"./model/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}.txt", "w") as f:
    #     f.write(np.array2string(np.ndarray(info["FD"])))

    envs.close()
    writer.close()
