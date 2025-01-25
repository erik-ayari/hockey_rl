import torch
from torch import nn

from gymnasium import spaces

class Critic(nn.Module):
    def __init__(
        self,
        state_space: spaces.Space,
        action_space: spaces.Box,
        num_layers: int,
        hidden_dim: int
    ):
        super(Critic, self).__init__()

        # Twin Network:
        self.critic1 = self._init_twin(
            state_space,
            action_space,
            num_layers,
            hidden_dim
        )

        self.critic2 = self._init_twin(
            state_space,
            action_space,
            num_layers,
            hidden_dim
        )

    def _init_twin(
        self,
        state_space: spaces.Space,
        action_space: spaces.Box,
        num_layers: int,
        hidden_dim: int
    ):
        # Observation -> Latent State Space
        layers = [
            nn.Linear(in_features=(state_space.shape[0] + action_space.shape[0]), out_features=hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            layers.append(nn.ReLU())

        # Latent State Space -> Q Value
        layers.append(nn.Linear(in_features=hidden_dim, out_features=1))

        return nn.Sequential(*layers)

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=-1)
        q1 = self.critic1(x)
        q2 = self.critic2(x)

        return q1, q2

