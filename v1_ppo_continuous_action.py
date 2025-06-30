
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import warnings
warnings.filterwarnings("ignore")
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import tyro
from tqdm import tqdm, trange
from collections import deque

from torch.utils.tensorboard import SummaryWriter
import argparse
parser = argparse.ArgumentParser()
from agent.ppo_agnet import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[: -len(".py")])
parser.add_argument("--seed", type=int, default=1,
                    help="seed of the experiment")
parser.add_argument("--torch_deterministic", default=True)
parser.add_argument("--cuda", default=True)
parser.add_argument("--track",default=False)
parser.add_argument("--wandb_project_name", type=str, default="cleanRL")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--capture_video",    default=False)
parser.add_argument("--save_model",    default=False)
parser.add_argument("--upload_model",    default=False)
parser.add_argument("--hf_entity", type=str, default="")

# Algorithm specific arguments
parser.add_argument("--env_id", type=str, default="Hopper-v4")
parser.add_argument("--total_timesteps", type=int, default=1000000)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=2048)
parser.add_argument("--anneal_lr",    default=True)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--num_minibatches", type=int, default=32)
parser.add_argument("--update_epochs", type=int, default=10)
parser.add_argument("--norm_adv",    default=True)
parser.add_argument("--clip_coef", type=float, default=0.2)
parser.add_argument("--clip_vloss",  default=True)
parser.add_argument("--ent_coef", type=float, default=0.0)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--target_kl", type=float, default=None)

# Runtime computed arguments
parser.add_argument("--batch_size", type=int, default=0)
parser.add_argument("--minibatch_size", type=int, default=0)
parser.add_argument("--num_iterations", type=int, default=0)
parser.add_argument("--run", type=int, default=1)




def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


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
        [make_env(args.env_id, i, args.capture_video, file_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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

    global_step_bar = trange(1, int(args.num_iterations + 1))

    metric = {
        "moving_avg_return": deque(maxlen=50),
        "moving_avg_length": deque(maxlen=50),
        "best_avg_return": -np.inf,
    }

    best_model_return = 0

    # for iteration in range(1, args.num_iterations + 1):
    for iteration in global_step_bar:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        metric["moving_avg_return"].append(info["episode"]["r"])
                        metric["moving_avg_length"].append(info["episode"]["l"])
                        break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if (iteration) % 1 == 0:
            global_step_bar.set_postfix(global_step=(global_step), avg_return_mean=np.mean(metric["moving_avg_return"]))
            logger.add_scalar("performace/moving_avg_return_mean", np.mean(metric["moving_avg_return"]), (global_step))
            logger.add_scalar("performace/moving_avg_return_std", np.std(metric["moving_avg_return"]), (global_step))
            logger.add_scalar("performace/moving_avg_length_mean", np.mean(metric["moving_avg_length"]), (global_step))
            logger.add_scalar("performace/moving_avg_length_std", np.std(metric["moving_avg_length"]), (global_step))
            # eval_return_mean = make_eval(actor, file_name, device, args)
            episodic_returns = evaluate(
                                agent,
                                make_env,
                                args.env_id,
                                eval_episodes=10,
                                run_name=f"{file_name}-eval",
                                Model=Agent,
                                device=device,
                                gamma=args.gamma,
                            )
            logger.add_scalar("performace/eval_return_mean", episodic_returns, (global_step))
            if episodic_returns > best_model_return:
                best_model_return = episodic_returns
                model_path = f"runs/{file_name}/{exp_tag}/model_checkpoint_best_model.pth"
                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")

    envs.close()
    logger.close()


if __name__ == "__main__":
    args = parser.parse_args()
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print(args)
    make_train(args)
    
    
    

