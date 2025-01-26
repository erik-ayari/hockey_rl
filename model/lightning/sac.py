from model.modules import Actor, Critic
from data import Experience, ReplayBuffer, RLDataset
from utils import EnvironmentType, AgentType, SplitActionSpace

from typing import Tuple, OrderedDict, List, Union
from math import prod

import numpy as np
import gymnasium as gym

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
        model_config: dict
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

        # Global variable that determines whether the populate function needs to reset the environment
        self.done = True

        # Warm Up Buffer
        self.populate(start_steps - 1, warm_up=True)

    def populate(self, steps: int, warm_up=False):
        # Reset Env
        if self.done:
            self.state, _ = self.env.reset()
        for _ in range(steps):
            # Sample Action
            # Either randomly (to increase variance/diversity in dataset compared to using untrained agent)
            if warm_up:
                action = self.action_space.sample()
            # Or sample from agent
            else:
                state_tensor = torch.tensor(self.state, dtype=torch.float).unsqueeze(0)
                with torch.no_grad():
                    action = self.actor.forward(state_tensor)[0][0].detach().numpy()
            # Collect consequences
            next_state, reward, done, truncated, _ = self.env.step(action)

            # We consider truncated to be also done
            done = done or truncated

            # Append to buffer
            experience = Experience(self.state, action, reward, done, next_state)
            self.buffer.append(experience)

            # Reset if necessary, otherwise next state becomes initial state
            if done:
                self.state, _ = self.env.reset()
            else:
                self.state = next_state.copy()
            
            self.done = done

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        self.populate(steps=1, warm_up=(self.current_epoch == 0))

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
        self.log("alpha_loss", alpha_loss, prog_bar=True)

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
        self.log("critic_loss", critic_loss, prog_bar=True)

        # Optimize Critic
        self.optimizers()[1].zero_grad()
        critic_loss.backward()
        self.optimizers()[1].step()

        # Calculate Actor Loss using its chosen actions and their q value
        policy_q_values = torch.cat(self.critic.forward(states, policy_actions), dim=1)
        policy_min_q = torch.min(policy_q_values, dim=1, keepdim=True)[0]
        actor_loss = (alpha * actions_log_probs - policy_min_q).mean()
        self.log("actor_loss", actor_loss, prog_bar=True)

        # Optimize Actor
        self.optimizers()[0].zero_grad()
        actor_loss.backward()
        self.optimizers()[0].step()

        # Soft Update the Critic Target
        self.critic_target.soft_update(self.critic)

    def validation_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
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
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                with torch.no_grad():
                    # For validation we consider the actor's predicted mode as its action
                    action = self.actor.forward(state_tensor, deterministic=True)[0].numpy()

                next_state, reward, done, truncated, info = self.env.step(action)

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
            batch_size=self.batch_size
        )
        return dataloader

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer_actor             = Adam(self.actor.parameters(), lr=self.actor_lr)
        optimizer_critic            = Adam(self.critic.parameters(), lr=self.critic_lr)
        optimizer_alpha             = Adam([self.log_alpha], lr=self.alpha_lr)
        return [optimizer_actor, optimizer_critic, optimizer_alpha]

