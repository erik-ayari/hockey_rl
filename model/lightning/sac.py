from model.modules import Actor, Critic
from data import Experience, ReplayBuffer, RLDataset, OpponentPool
from utils import EnvironmentType, AgentType, SplitActionSpace
from foreign_agents import MPODestilledAgent, TDMPC2DestilledAgent
from hockey.hockey_env import BasicOpponent

from typing import Tuple, OrderedDict, List, Union
from math import prod
from tqdm import tqdm

import numpy as np
import gymnasium as gym

import trueskill

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl

class SoftActorCritic(pl.LightningModule):
    def __init__(
        self,
        env,
        environment_type: EnvironmentType,
        agent_type: AgentType,
        split_action_space: SplitActionSpace,
        model_config: dict,
        resume = False
    ):
        super(SoftActorCritic, self).__init__()

        self.save_hyperparameters(model_config)

        # No Automatic Optimization because we have separate Optimizers for Actor, Critic, Alpha
        self.automatic_optimization = False

        # Environment parameters
        self.env                = env
        self.environment_type   = environment_type
        self.agent_type         = agent_type
        self.split_action_space = split_action_space
        # Split action space if we actually only use the first half (like in checkpoint 2)
        if self.split_action_space == SplitActionSpace.SPLIT:
            # Calculate the midpoint
            midpoint = env.action_space.shape[0] // 2

            # Slice the minima and maxima
            low = env.action_space.low[:midpoint]
            high = env.action_space.high[:midpoint]

            # Define halved action space
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=env.action_space.dtype)
        else:
            self.action_space   = env.action_space
        
        self.state_dim          = prod(env.observation_space.shape)
        self.action_dim         = prod(env.action_space.shape)
        # Again, halve action_dim if action space should be splitted
        self.action_dim = self.action_dim // 2 if self.split_action_space == SplitActionSpace.SPLIT else self.action_dim
        self.gamma              = model_config.get('gamma', 0.99)

        # Actor hyperparameters
        actor_config            = model_config.get('actor', {})
        self.actor_num_layers   = actor_config.get('num_layers', 1)
        self.actor_hidden_dim   = actor_config.get('hidden_dim', 256)
        self.actor_lr           = actor_config.get('lr', 3e-4)
        self.actor = Actor(
            state_dim   = self.state_dim,
            action_dim  = self.action_dim,
            num_layers  = self.actor_num_layers,
            hidden_dim  = self.actor_hidden_dim
        )

        # Critic hyperparameters
        critic_config = model_config.get('critic', {})
        self.critic_num_layers = critic_config.get('num_layers', 1)
        self.critic_hidden_dim = critic_config.get('hidden_dim', 256)
        self.critic_lr = critic_config.get('lr', 3e-4)
        self.critic_tau = critic_config.get('tau', 0.005)
        # Twin Critics:
        self.critic = Critic(
            state_dim   = self.state_dim,
            action_dim  = self.action_dim,
            num_layers  = self.critic_num_layers,
            hidden_dim  = self.critic_hidden_dim
        )

        # Target Critics
        self.critic_target = Critic(
            state_dim   = self.state_dim,
            action_dim  = self.action_dim,
            num_layers  = self.critic_num_layers,
            hidden_dim  = self.critic_hidden_dim,
            tau         = self.critic_tau
        )
        # Initialize to the critics parameters
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Alpha Hyperparameters
        alpha_config = model_config.get('alpha', {})
        self.alpha_lr = alpha_config.get('lr', 3e-4)
        # Entropy Coefficient initialized
        initial_log_alpha = alpha_config.get('log_init', 1)
        self.log_alpha = nn.Parameter(torch.ones(initial_log_alpha, requires_grad=True))

        # If no target entropy is set, default to original paper's heuristic
        self.target_entropy = model_config.get('target_entropy', -self.action_dim)
        if self.target_entropy == None:
            self.target_entropy = -self.action_dim

        # Technically training, not model parameters, but for simplicity:
        self.batch_size = model_config.get('batch_size', 256)
        self.validation_length = model_config.get('validation_length', 20)

        # Replay Buffer
        self.buffer = ReplayBuffer(model_config.get('replay_size', 1_000_000))
        start_steps = model_config.get('start_steps', 100)

        self.opponent_observations = model_config.get('opponent_observations', False)

        # Global variable that determines whether the populate function needs to reset the environment
        self.done = True

        pool_config = model_config.get('pool', None)
        if pool_config == None:
            self.use_pool = False
        else:
            self.use_pool           = pool_config["use_pool"]
            self.snapshot_interval  = pool_config["snapshot_interval"]

            checkpoints_mpo = pool_config.get("checkpoints_mpo", [])
            checkpoints_tdmpc2 = pool_config.get("checkpoints_tdmpc2", [])
            checkpoints_sac = pool_config.get("checkpoints_sac", [])
            
            foreign_agents = {}
            for mpo_path in checkpoints_mpo:
                name = mpo_path
                mpo_path = f"foreign_agents/checkpoints/{mpo_path}.pth"
                mpo = MPODestilledAgent(mpo_path)
                foreign_agents[name] = mpo
            for tdmpc2_path in checkpoints_tdmpc2:
                name = tdmpc2_path
                tdmpc2_path = f"foreign_agents/checkpoints/{tdmpc2_path}.pth"
                tdmpc2 = TDMPC2DestilledAgent(tdmpc2_path)
                foreign_agents[name] = tdmpc2
            for sac_path in checkpoints_sac:
                name = sac_path
                sac_path = f"foreign_agents/checkpoints/{sac_path}.ckpt"
                sac = Actor(state_dim=18, action_dim=4, num_layers=2, hidden_dim=256)
                sac.load_checkpoint(sac_path)
                foreign_agents[name] = sac

            self.opponent_pool      = OpponentPool(
                actor_params = {
                    "state_dim" : self.state_dim,
                    "action_dim": self.action_dim,
                    "num_layers": self.actor_num_layers,
                    "hidden_dim": self.actor_hidden_dim
                },
                foreign_agents=foreign_agents,
                weighting=pool_config.get("weighting", []),
                device=self.device
            )
            self.snapshot_interval_steps = 0
            self.pool_games_per_opponent = pool_config["games_per_opponent"]

        self.steps_per_epoch = model_config.get('steps_per_epoch', 1000)

        # Bootstrap with Strong Basic Opponent
        self.bootstrap_agent = BasicOpponent(weak=False)
        self.bootstrap_steps = model_config.get('bootstrap_steps', self.steps_per_epoch)

        self.resume = resume

        # Warm Up Buffer
        #self.populate(warm_up=True)

    def on_train_start(self):
        self.warm_up_required = True

    def on_train_epoch_start(self):
        if self.current_epoch % self.steps_per_epoch == 0:
            self.populate(warm_up=self.warm_up_required)
            self.warm_up_required = False

    def populate(self, warm_up=False):
        # Reset Env
        if self.done:
            self.state, _ = self.env.reset()
            if self.use_pool:
                self.opponent_pool.sample_opponent()
        if self.environment_type == EnvironmentType.GAME:
            self.state_opponent = self.env.obs_agent_two()
        # Pick opponent if pool is used
        steps = self.bootstrap_steps if warm_up else self.steps_per_epoch
        for _ in range(steps):
            # Sample Action
            # Bootstrap with Strong Opponent
            if warm_up and not self.resume:
                action = self.bootstrap_agent.act(self.state)
            # Or sample from agent
            else:
                state_tensor = torch.tensor(self.state, dtype=torch.float).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self.actor.forward(state_tensor)[0][0].detach().cpu().numpy()

            if self.use_pool:
                action_opponent = self.opponent_pool.act_opponent(self.state_opponent)
                action_step = np.concatenate((action, action_opponent))
            elif self.split_action_space == SplitActionSpace.SPLIT:
                action_step = np.concatenate((action, np.zeros_like(action)))
            else:
                action_step = action.copy()
            # Collect consequences
            next_state, reward, done, truncated, info = self.env.step(action_step)
            if self.environment_type == EnvironmentType.GAME:
                action_opponent = info["action_player2"]
                next_state_opponent = self.env.obs_agent_two()
                reward_opponent = self.env.get_reward_agent_two(self.env.get_info_agent_two())

            # We consider truncated to be also done
            done = done or truncated

            # Append to buffer
            experience = Experience(self.state, action, reward, done, next_state)
            self.buffer.append(experience)

            # Append opponent obs to buffer
            if self.opponent_observations:
                experience_opponent = Experience(self.state_opponent, action_opponent, reward_opponent, done, next_state_opponent)

            # Reset if necessary, otherwise next state becomes initial state
            if done:
                self.opponent_pool.udpate_rating(info['winner'])
                self.state, _ = self.env.reset()
                if self.environment_type == EnvironmentType.GAME:
                    self.state_opponent = self.env.obs_agent_two()
                # Pick new opponent
                if self.use_pool:
                    self.opponent_pool.sample_opponent()
            else:
                self.state = next_state.copy()
                if self.environment_type == EnvironmentType.GAME:
                    self.state_opponent = next_state_opponent.copy()
            
            self.done = done

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        states, actions, rewards, dones, next_states = batch

        # Convert dones to float so we can calculate with it
        dones = dones.float()
        
        # Get actions for interaction, probabilities for optimization
        policy_actions, actions_log_probs = self.actor.forward(states)

        # Separate Batch Dimension
        actions_log_probs = actions_log_probs.unsqueeze(1)

        # We will use this alpha in the calculation of the other losses, so that is not affected
        alpha = self.log_alpha.exp().detach()

        # Loss of Alpha
        alpha_loss = -(self.log_alpha * (actions_log_probs + self.target_entropy).detach()).mean()
        self.log("alpha_loss", alpha_loss, on_epoch=True)

        # Optimize Alpha
        self.optimizers()[2].zero_grad()
        alpha_loss.backward()
        self.optimizers()[2].step()

        # This is required to get the recursive nature of the Q function
        # However, we do not want to propagate the gradient!
        with torch.no_grad():
            # Get actions and log probabilities of the next state according to policy
            next_actions, next_actions_log_probs = self.actor.forward(next_states)
            # Compute the estimated Q value as the minimum of the two critic targets
            next_q_values = torch.cat(self.critic_target.forward(next_states, next_actions), dim=-1)
            next_q_values = torch.min(next_q_values, dim=1, keepdim=True)[0]
            # (Negative) Entropy Term weighted by alpha
            next_q_values = next_q_values - alpha * next_actions_log_probs.unsqueeze(1)
            # TD Error (only propagated to future states if we were not done)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        # We train the critic with the actions that were actually taken
        # Get current Q estimates by critic
        current_q_values = self.critic.forward(states, actions)

        # Critic Loss calculation
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        self.log("critic_loss", critic_loss, on_epoch=True)

        # Optimize Critic
        self.optimizers()[1].zero_grad()
        critic_loss.backward()
        self.optimizers()[1].step()

        # Calculate Actor Loss using its chosen actions and their q value
        policy_q_values = torch.cat(self.critic.forward(states, policy_actions), dim=1)
        policy_min_q = torch.min(policy_q_values, dim=1, keepdim=True)[0]
        actor_loss = (alpha * actions_log_probs - policy_min_q).mean()
        self.log("actor_loss", actor_loss, on_epoch=True)

        # Optimize Actor
        self.optimizers()[0].zero_grad()
        actor_loss.backward()
        self.optimizers()[0].step()

        # Soft Update the Critic Target
        self.critic_target.soft_update(self.critic)

        if self.use_pool:
            self.snapshot_interval_steps += 1
            if self.snapshot_interval_steps % self.snapshot_interval == 0:
                self.opponent_pool.add_snapshot(self.actor)

    def regular_validation(self):
        # Track Wins and Draws in case of game
        num_wins = 0
        num_draws = 0
        # Track average cumulative reward otherwise
        cumulative_rewards = []
        
        for episode in range(self.validation_length):
            state, _ = self.env.reset()
            # Track reward if no game validation
            cumulative_reward = 0.0
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    # For validation we consider the actor's predicted mode as its action
                    action = self.actor.forward(state_tensor, deterministic=True)[0].cpu().numpy()

                if self.split_action_space == SplitActionSpace.SPLIT:
                    action_step = np.concatenate((action, np.zeros_like(action)))
                else:
                    action_step = action.copy()

                next_state, reward, done, truncated, info = self.env.step(action_step)

                if self.environment_type != EnvironmentType.GAME:
                    cumulative_reward += reward

                # Again, we consider timeouts as done
                done = done or truncated

                # If done, track statistic
                if done:
                    # In case of game, track winner
                    if self.environment_type == EnvironmentType.GAME:
                        if info['winner'] == 1:
                            num_wins += 1
                        elif info['winner'] == 0:
                            num_draws += 1
                    # Otherwise track cum. reward
                    else:
                        cumulative_rewards.append(cumulative_reward)
                state = next_state.copy()
        
        if self.environment_type == EnvironmentType.GAME:
            win_rate = num_wins / self.validation_length
            draw_rate = num_draws / self.validation_length

            self.log("val_win_rate", win_rate, prog_bar=True)
            self.log("val_draw_rate", draw_rate, prog_bar=True)
        else:
            self.log("val_cumulative_reward", np.array(cumulative_rewards).mean(), prog_bar=True)

        self.done = True

    def play_1v1(self):
        state, _ = self.env.reset()
        state_opponent = self.env.obs_agent_two()
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
            with torch.no_grad():
                action = self.actor.forward(state_tensor, deterministic=True)[0].cpu().numpy()
                action_opponent = self.opponent_pool.act_opponent(state_opponent)

                action_step = np.hstack((action, action_opponent))

            state, _, done, truncated, info = self.env.step(action_step)

            state_opponent = self.env.obs_agent_two()

            done = (done or truncated)
        
        self.done = True
        
        return info['winner']

    def tournament_validation(self):
        opponent_ratings = {}
        opponnent_mus = []
        opponent_sigmas = []
        model_rating = trueskill.Rating()

        wins_weak       = 0.0
        wins_strong     = 0.0
        wins_mpo        = 0.0
        draws_weak      = 0.0
        draws_strong    = 0.0
        draws_mpo       = 0.0

        for opponent_idx in range(len(self.opponent_pool)):
            opponent_rating = trueskill.Rating()
            self.opponent_pool.set_opponent(opponent_idx)

            for _ in range(self.pool_games_per_opponent):
                outcome = self.play_1v1()
                model_rating = self.opponent_pool.udpate_rating(model_rating, opponent_idx, outcome)

                if opponent_idx < 3:
                    if opponent_idx == 0:
                        if outcome == 1:
                            wins_weak += 1
                        elif outcome == 0:
                            draws_weak += 1
                    elif opponent_idx == 1:
                        if outcome == 1:
                            wins_strong += 1
                        elif outcome == 0:
                            draws_strong += 1
                    else:
                        if outcome == 1:
                            wins_mpo += 1
                        elif outcome == 0:
                            draws_mpo += 1

            opponnent_mus.append(self.opponent_pool.get_rating(opponent_idx))
            opponent_sigmas.append(self.opponent_pool.get_rating(opponent_idx, sigma=True))
        
        model_mu = model_rating.mu
        model_sigma = model_rating.sigma
        count = 0.0
        for opp_mu in opponnent_mus:
            if model_mu > opp_mu:
                count += 1.0
            elif model_mu == opp_mu:
                count += 0.5  # Counting ties as half
        percentile = count / len(opponnent_mus) if opponnent_mus else 0.0

        win_rate_weak       = wins_weak     / self.pool_games_per_opponent
        win_rate_strong     = wins_strong   / self.pool_games_per_opponent
        win_rate_mpo        = wins_mpo      / self.pool_games_per_opponent

        draw_rate_weak      = draws_weak    / self.pool_games_per_opponent
        draw_rate_strong    = draws_strong  / self.pool_games_per_opponent
        draw_rate_mpo       = draws_mpo     / self.pool_games_per_opponent

        self.log("win_rate_weak",       win_rate_weak,      prog_bar=True)
        self.log("win_rate_strong",     win_rate_strong,    prog_bar=True)
        self.log("win_rate_mpo",        win_rate_mpo,       prog_bar=True)
        self.log("draw_rate_weak",      draw_rate_weak)
        self.log("draw_rate_strong",    draw_rate_strong)
        self.log("draw_rate_mpo",       draw_rate_mpo)

        self.log("val_percentile",          percentile,     prog_bar=True)
        self.log("val_model_rating_mu",     model_mu)
        self.log("val_model_rating_sigma",  model_sigma)

        opponnent_mus = np.array(opponnent_mus)
        opponent_sigmas = np.array(opponent_sigmas)

        self.log("val_opp_mu",          opponnent_mus.mean())
        self.log("val_opp_sigma",       opponent_sigmas.mean())

        self.log("val_weak-opp_mu",         opponnent_mus[0])
        self.log("val_strong-opp_mu",       opponnent_mus[1])
        self.log("val_mpo-opp_mu",          opponnent_mus[2])
        self.log("val_mpo2-opp_mu",          opponnent_mus[3])
        self.log("val_weak-opp_sigma",      opponent_sigmas[0])
        self.log("val_strong-opp_sigma",    opponent_sigmas[1])
        self.log("val_mpo-opp_sigma",       opponent_sigmas[2])
        self.log("val_mpo2-opp_sigma",       opponent_sigmas[3])

        if len(opponnent_mus) > 4:
            self.log("val_self-opp_mu", np.array(opponnent_mus)[4:].mean(), prog_bar=True)
            self.log("val_self-opp_sigma", np.array(opponent_sigmas)[4:].mean(), prog_bar=True)

    def log_ratings(self):
        ratings, model_rating = self.opponent_pool.get_ratings()
        for key in ratings["basic"]:
            mu = ratings["basic"][key].mu
            sigma = ratings["basic"][key].sigma
            self.log(f"{key}_mu", mu, prog_bar=True)
            self.log(f"{key}_sigma", sigma)
        snapshot_mus = []
        snapshot_sigmas = []
        for key in ratings["snapshots"]:
            snapshot_mus.append(ratings["snapshots"][key].mu)
            snapshot_sigmas.append(ratings["snapshots"][key].sigma)
        
        if len(snapshot_mus) > 0:
            snapshot_mus = np.array(snapshot_mus)
            snapshot_sigmas = np.array(snapshot_sigmas)
            self.log("snapshots_mu", snapshot_mus.mean(), prog_bar=True)
            self.log("snapshots_mu_max", snapshot_mus.max(), prog_bar=True)
            self.log("snapshots_sigma", snapshot_sigmas.mean())
        
        if self.opponent_pool.foreign_agents_exist:
            foreign_mus = []
            foreign_sigmas = []
            for key in ratings["foreign"]:
                foreign_mus.append(ratings["foreign"][key].mu)
                foreign_sigmas.append(ratings["foreign"][key].sigma)
            foreign_mus = np.array(foreign_mus)
            foreign_sigmas = np.array(foreign_sigmas)
            self.log("foreign_mu", foreign_mus.mean(), prog_bar=True)
            self.log("foreign_mu_max", foreign_mus.max(), prog_bar=True)
            self.log("foreign_sigma", foreign_sigmas.mean())

        self.log("model_mu", model_rating.mu, prog_bar=True)
        self.log("model_sigma", model_rating.sigma)

    def validation_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        if self.use_pool:
            self.log_ratings()
            #self.tournament_validation()
        else:
            self.regular_validation()

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def val_dataloader(self) -> DataLoader:
        return DataLoader([0], batch_size=1)

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            pin_memory=True
        )
        return dataloader

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer_actor             = Adam(self.actor.parameters(), lr=self.actor_lr)
        optimizer_critic            = Adam(self.critic.parameters(), lr=self.critic_lr)
        optimizer_alpha             = Adam([self.log_alpha], lr=self.alpha_lr)
        return [optimizer_actor, optimizer_critic, optimizer_alpha]

