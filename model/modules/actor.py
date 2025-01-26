import torch
from torch import nn
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_layers: int,
        hidden_dim: int,
        epsilon: float = 1e-6
    ):
        super(Actor, self).__init__()

        self.epsilon = epsilon

        # Observation -> Latent State Space
        layers = [
            nn.Flatten(),
            nn.Linear(in_features=state_dim, out_features=hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            layers.append(nn.ReLU())

        self.actor_latent = nn.Sequential(*layers)


        # Latent State Space -> Action Space as Diagonal Normal
        self.actor_mu = nn.Linear(in_features=hidden_dim, out_features=action_dim)
        self.actor_log_std = nn.Linear(in_features=hidden_dim, out_features=action_dim)

    def _sample(self, mu, log_std):
        normal_dist = Normal(mu, log_std.exp())

        # Sample with reparametrization trick
        z = normal_dist.rsample()
        
        # Calculate log probability along each (independent) dimension
        log_prob = normal_dist.log_prob(z)

        # Calculate log probability of diagonal normal
        log_prob = normal_dist.log_prob(z).sum(dim=-1)

        return z, log_prob

    def _squash(self, action, log_prob):
        action = torch.tanh(action)
        # Squash Correction from original paper
        log_prob -= torch.sum(torch.log(1 - action**2 + self.epsilon), dim=1)
        return action, log_prob

    def forward(self, x, deterministic=False):
        latent = self.actor_latent(x)

        mu = self.actor_mu(latent)
        
        if deterministic:
            return mu

        log_std = self.actor_log_std(latent).clamp(min=-20, max=2)

        action, log_prob = self._sample(mu, log_std)

        # Squash to [-1, 1]
        action, log_prob = self._squash(action, log_prob)

        return action, log_prob

