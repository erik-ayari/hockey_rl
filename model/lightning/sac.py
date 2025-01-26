from model.modules import Actor, Critic
from data import Experience, ReplayBuffer, RLDataset

from typing import Tuple, OrderedDict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl

class SoftActorCritic(pl.LightningModule):
    def __init__(
        self,
        # Environment Related Parameters
        env,
        gamma               : float = 0.99,
        # Network Architecture
        #  Actor
        actor_num_layers    : int = 1,
        actor_hidden_dim    : int = 256,
        #  Critic
        critic_num_layers   : int = 1,
        critic_hidden_dim   : int = 256,
        # Optimizer Parameters
        actor_lr            : float = 3e-4,
        critic_lr           : float = 3e-4,
        alpha_lr            : float = 3e-4,
        tau                 : float = 0.005,
        # Training Procedure
        batch_size          : int   = 256,
        replay_size         : float = 1_000_000,
        start_steps         : int   = 100,
        # Validation Procedure
        val_episodes        : int = 20
    ):
        super(SoftActorCritic, self).__init__()

        self.save_hyperparameters()

        # No Automatic Optimization because we have separate Optimizers for Actor, Critic, Alpha
        self.automatic_optimization = False

        # Environment related parameters
        self.env = env
        self.state_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.gamma = gamma

        # Training Procedure Parameters
        self.tau = tau
        self.batch_size = batch_size

        # Validation Procedure Parameters
        self.val_episodes = val_episodes

        # Actor
        self.actor = Actor(
            self.state_space,
            self.action_space,
            actor_num_layers,
            actor_hidden_dim
        )
        self.actor_lr = actor_lr

        # Twin Critis
        self.critic = Critic(
            self.state_space,
            self.action_space,
            critic_num_layers,
            critic_hidden_dim
        )
        self.critic_lr = critic_lr

        # Twin Critic Targets (initialized to the same parameters)
        self.critic_target = Critic(
            self.state_space,
            self.action_space,
            critic_num_layers,
            critic_hidden_dim
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy Coefficient (learned)
        self.log_alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.alpha_lr = alpha_lr
        self.target_entropy = -self.action_space.shape[0]

        # Replay Buffer
        self.buffer = ReplayBuffer(replay_size)

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
            next_state, reward, done, _, _ = self.env.step(action)

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

        # Update Target Network
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

    def validation_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        num_wins = 0
        num_stales = 0
        for episode in range(self.val_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                with torch.no_grad():
                    action = self.actor.forward(state_tensor, deterministic=True)[0].numpy()

                next_state, reward, done, _, info = self.env.step(action)

                if done:
                    if info['winner'] == 1:
                        num_wins += 1
                    elif info['winner'] == 0:
                        num_stales += 1
                state = next_state.copy()
        
        win_rate = num_wins / self.val_episodes
        stale_rate = num_stales / self.val_episodes

        self.log("val_win_rate", win_rate, prog_bar=True)
        self.log("val_stale_rate", stale_rate, prog_bar=True)

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

