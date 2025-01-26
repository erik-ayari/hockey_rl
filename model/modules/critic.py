import torch
from torch import nn

class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_layers: int,
        hidden_dim: int,
        tau: float = None
    ):
        super(Critic, self).__init__()

        # Twin Network:
        self.critic1 = self._init_twin(
            state_dim,
            action_dim,
            num_layers,
            hidden_dim
        )

        self.critic2 = self._init_twin(
            state_dim,
            action_dim,
            num_layers,
            hidden_dim
        )

        self.tau = tau

    def _init_twin(
        self,
        state_dim: int,
        action_dim: int,
        num_layers: int,
        hidden_dim: int
    ):
        # Observation -> Latent State Space
        layers = [
            nn.Linear(in_features=(state_dim + action_dim), out_features=hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            layers.append(nn.ReLU())

        # Latent State Space -> Q Value
        layers.append(nn.Linear(in_features=hidden_dim, out_features=1))

        return nn.Sequential(*layers)

    def soft_update(self, critic):
        if self.tau == None:
            raise ValueError("Tau was not specified. Likely, I am not a target network!")
        with torch.no_grad():
            for param, target_param in zip(critic.parameters(), self.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=-1)
        q1 = self.critic1(x)
        q2 = self.critic2(x)

        return q1, q2

