from torch import nn
from torch.distributions import Normal

from gymnasium import spaces

class Actor(nn.Module):
    def __init__(
        self,
        state_space: spaces.Space,
        action_space: spaces.Box,
        num_layers: int,
        hidden_dim: int
    ):
        super(Actor, self).__init__()

        # Observation -> Latent State Space
        layers = [
            nn.Linear(in_features=state_space.shape[0], out_features=hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            layers.append(nn.ReLU())

        self.actor_latent = nn.Sequential(*layers)


        # Latent State Space -> Action Space as Diagonal Normal
        self.actor_mu = nn.Linear(in_features=hidden_dim, out_features=action_space.shape[0])
        self.actor_log_std = nn.Linear(in_features=hidden_dim, out_features=action_space.shape[0])

    def _sample(self, mu, log_std):
        normal_dist = Normal(mu, log_std.exp())

        # Sample with reparametrization trick
        z = normal_dist.rsample()
        
        # Calculate log probability along each (independent) dimension
        log_prob = normal_dist.log_prob(z)

        # Calculate log probability of diagonal normal
        log_prob = normal_dist.log_prob(z).sum(dim=-1)

        return z, log_prob

    def forward(self, x, deterministic=False):
        latent = self.actor_latent(x)

        mu = self.actor_mu(latent)
        
        if deterministic:
            return mu

        log_std = self.actor_log_std(latent)

        return self._sample(mu, log_std)

