# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import warnings
warnings.filterwarnings("ignore")
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from collections import deque
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from agent.sac_agent import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"))
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--torch_deterministic", action="store_true", default=True)
parser.add_argument("--cuda", default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
parser.add_argument("--track", default=False, help="if toggled, this experiment will be tracked with Weights and Biases")
parser.add_argument("--wandb_project_name", type=str, default="cleanRL", help="the wandb's project name")
parser.add_argument("--wandb_entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--capture_video", action="store_true", default=False, help="whether to capture videos of the agent performances")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_starts", type=int, default=5e3)
parser.add_argument("--policy_lr", type=float, default=3e-4)
parser.add_argument("--q_lr", type=float, default=1e-3)
parser.add_argument("--policy_frequency", type=int, default=2)
parser.add_argument("--target_network_frequency", type=int, default=1)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--autotune", type=bool, default=True)

parser.add_argument("--eval_seed", type=int, default=2025)
parser.add_argument("--env_id", type=str, default="Walker2d-v4")
parser.add_argument("--total_timesteps", type=int, default=int(1e6))
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--buffer_size", type=int, default=int(1e6))
parser.add_argument("--eval_num", type=int, default=int(10))
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--n_steps", type=int, default=int(3))


def make_train(args):
    file_name = args.exp_name
    if not os.path.exists(f"runs/{file_name}"):
        os.makedirs(f"runs/{file_name}")

    exp_tag = f"{args.env_id}_{args.exp_name}_{args.seed}_{args.run}"
    logger = SummaryWriter(f"runs/{file_name}/{exp_tag}")
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, file_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)


    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32

    metric = {
        "moving_avg_return": deque(maxlen=50),
        "moving_avg_length": deque(maxlen=50),
        "best_avg_return": -np.inf,
    }

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    global_step_bar = trange(1, int(args.total_timesteps + 1), desc=f"runs/{file_name}/{exp_tag}")

    obs, _ = envs.reset(seed=args.seed)

    best_model_return = 0

    for global_step in global_step_bar:

        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    metric["moving_avg_return"].append(info["episode"]["r"])
                    metric["moving_avg_length"].append(info["episode"]["l"])
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if (global_step) % 10000 == 0:
                global_step_bar.set_postfix(global_step=(global_step), avg_return_mean=np.mean(metric["moving_avg_return"]))
                logger.add_scalar("performace/moving_avg_return_mean", np.mean(metric["moving_avg_return"]), (global_step))
                logger.add_scalar("performace/moving_avg_return_std", np.std(metric["moving_avg_return"]), (global_step))
                logger.add_scalar("performace/moving_avg_length_mean", np.mean(metric["moving_avg_length"]), (global_step))
                logger.add_scalar("performace/moving_avg_length_std", np.std(metric["moving_avg_length"]), (global_step))
                eval_return_mean = make_eval(actor, file_name, device, args)
                logger.add_scalar("performace/eval_return_mean", eval_return_mean, (global_step))
                if eval_return_mean > best_model_return:
                    best_model_return = eval_return_mean
                    save_actor(actor, path=f"runs/{file_name}/{exp_tag}/model_checkpoint_best_model.pth")
            if (global_step) % 100000 == 0:
                save_actor(actor, path=f"runs/{file_name}/{exp_tag}/model_checkpoint_{global_step}.pth")

    envs.close()
    logger.close()


def make_eval(actor, file_name, device, args):
    metric = {
        "eval_num": int(0),
        "eval_return": deque(maxlen=args.eval_num),
        "eval_length": deque(maxlen=args.eval_num)
    }
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, file_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs, _ = envs.reset()
    while metric["eval_num"] < args.eval_num:
        with torch.no_grad():
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    metric["eval_num"] += 1
                    metric["eval_return"].append(info["episode"]["r"])
                    metric["eval_length"].append(info["episode"]["l"])
        
        obs = next_obs
    print(f'eval_return {np.array(metric["eval_return"]).mean()}')
    return np.array(metric["eval_return"]).mean()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    make_train(args)

    