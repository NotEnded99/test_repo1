
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from agent.dynamic import StableDynamicsModel

parser = argparse.ArgumentParser()
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
parser.add_argument("--env_id", type=str, default="Hopper-v4") # Walker2d
parser.add_argument("--total_timesteps", type=int, default=20000)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--eval_num", type=int, default=int(1))
parser.add_argument("--run", type=int, default=1)


def make_test(args):
    file_name = "v2_sac_continuous_action"

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
    
    # load actor
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    envs.single_observation_space.dtype = np.float32
    load_actor(actor, path=f"/home/mzm/RL_codes/BreSTL_clean/runs/v2_sac_continuous_action/Hopper-v4_v2_sac_continuous_action_2025_2/model_checkpoint_best_model.pth")
    
    # load dynamics model
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    # print(state_dim, action_dim)
    HIDDEN_SIZE = 256
    dynamic_model = StableDynamicsModel(state_dim, action_dim, HIDDEN_SIZE, device=device).to(device)
    dynamic_model.load_model("Hopper_stable_dynamics_model_20250426_144632.pth")

    # load buffer
    save_path = '/data/mzm/transitions/Hopper-v3_dataset-v0.pt'
    data = torch.load(save_path)
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    next_states = data['next_states']
    deltas = next_states - states
    state_mean, state_std = states.mean(0), states.std(0) + 1e-6
    action_mean, action_std = actions.mean(0), actions.std(0) + 1e-6
    delta_mean, delta_std = deltas.mean(0), deltas.std(0) + 1e-6
    reward_mean, reward_std = rewards.mean(0), rewards.std(0) + 1e-6
    dynamic_model.set_normalizer(state_mean, state_std, action_mean, action_std, delta_mean, delta_std, reward_mean, reward_std)

    
    eval_return_mean = make_eval(actor, dynamic_model, file_name, device, args)
    print("performace/eval_return_mean:", eval_return_mean)


def make_eval(actor, dynamic_model, file_name, device, args):
    
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

        with torch.no_grad():
            state_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            action_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
            # print(state_tensor.size(), action_tensor.size())
            next_state_pred, reward_pred = dynamic_model.predict(state_tensor, 
                                                                 action_tensor, 
                                                                 deterministic=True)
        next_obs, _, _, _, infos = envs.step(actions)

        print(next_obs.squeeze(0)[0:2], next_state_pred.squeeze(0).cpu().numpy()[0:2])

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
    # print(args)
    make_test(args)

    