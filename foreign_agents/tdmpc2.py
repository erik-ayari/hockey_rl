from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

"""
That is a compact Tdmpc2 agent that can be used for hockey inference.

Author: Daniel Flat
"""

def log_std_func(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)

def gaussian_logprob_func(eps, log_std):
    """Compute Gaussian log probability."""
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385175704956
    return log_prob.sum(-1, keepdim = True)

def squash_func(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim = True)
    return mu, pi, log_pi

class NormedLinear(nn.Module):
    """
    Step 01: Linear Layer
    Step 02 (Optional): Dropout
    Step 03: LayerNorm
    Step 04: Activation Function (Mish and SimNorm are supported)
    """

    def __init__(self, in_features: int, out_features: int, activation_function: str, bias: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias = bias)
        self.dropout = nn.Dropout(dropout, inplace = False) if dropout else None
        self.layer_norm = nn.LayerNorm(out_features)

        if activation_function == "Mish":
            self.activation_function = nn.Mish(inplace = False)
        elif activation_function == "SimNorm":
            self.activation_function = SimNorm()
        else:
            raise NotImplementedError(f"Activation function {activation_function} not implemented.")

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.activation_function(x)
        return x

class SimNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, simnorm_dim = 8):
        shape = x.shape
        x = x.view(*shape[:-1], -1, simnorm_dim)
        x = F.softmax(x, dim = -1)
        return x.view(*shape)

TDMPC2 = {
    "DISCOUNT": 0.99,
    "CONSISTENCY_COEF": 20,  # factor for the consistency loss
    "REWARD_COEF": 0.1,  # factor for the reward loss
    "Q_COEF": 0.1,  # factor for the Q loss
    "ENTROPY_COEF": 1e-4,
    "ENC_LR_SCALE": 0.3,

    "HORIZON": 1,  # How many steps do we want to consider while doing predictions

    "MMPI_ITERATIONS": 1,  # How many iterations of MPPI should we use for planning
    "NUM_TRAJECTORIES": 8,
    "NUM_SAMPLES": 256,
    "NUM_ELITES": 64,
    "MIN_STD": 0.05,
    "MAX_STD": 2,
    "TEMPERATURE": 0.5,
    "LATENT_SIZE": 64,
    "LOG_STD_MIN": -10,
    "LOG_STD_MAX": 2,
    # which checkpoint should be used for the TD-MPC-2 Hockey agent?
}


class EncoderNet(nn.Module):
    def __init__(self, state_size: int, latent_size: int):
        super().__init__()

        self.encoder_net = nn.Sequential(
            NormedLinear(in_features=state_size, out_features=latent_size, activation_function="Mish"),
            NormedLinear(in_features=latent_size, out_features=latent_size, activation_function="SimNorm"),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encoding the state into a latent space because the model can handle with its one latent space better than the actual one.
        """
        latent_state = self.encoder_net(state)
        return latent_state


# TODO
class DynamicsNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.encoder_net = nn.Sequential(
            NormedLinear(in_features = latent_size + action_size, out_features = latent_size,
                         activation_function = "Mish"),
            # NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "Mish"),
            NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "SimNorm"),
        )

    def forward(self, latent_state: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Predict the next latent state given the current latent state and the action.
        """
        input = torch.cat([latent_state, latent_action], dim = -1)
        next_latent_state = self.encoder_net(input)
        return next_latent_state


class RewardNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.encoder_net = nn.Sequential(
            NormedLinear(in_features = latent_size + action_size, out_features = latent_size,
                         activation_function = "Mish"),
            # NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "Mish"),
            nn.Linear(latent_size, 1, bias = False),
        )

    def forward(self, latent_state: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Predict the reward given the current latent state and the action.
        """
        input = torch.cat([latent_state, latent_action], dim = -1)
        latent_state = self.encoder_net(input)
        return latent_state


class ActorNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.actor_net = nn.Sequential(
            NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "Mish"),
            NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "Mish"),
            nn.Linear(latent_size, 2 * action_size, bias = True),
        )  # mean, std

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Predict the mean and log_std of the action distribution given the latent state.
        """
        output = self.actor_net(latent_state)
        return output


class CriticNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.critic_net = nn.Sequential(
            NormedLinear(in_features = latent_size + action_size, out_features = latent_size, dropout = 0.01,
                         activation_function = "Mish"),
            NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "Mish"),
            nn.Linear(latent_size, 1, bias = True))

    def forward(self, latent_state: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Predict the q-value given the latent state and the action.
        THIS IS JUST AN ESTIMATE, NOT THE REAL Q-VALUE.
        """
        input = torch.cat([latent_state, latent_action], dim = -1)
        q_value = self.critic_net(input)
        return q_value


class TDMPC2DestilledAgent(nn.Module):
    def __init__(self, checkpoint_name: str, device: torch.device = torch.device("cpu"), eval=True):
        nn.Module.__init__(self)

        self.device = device
        self.state_size = 18
        self.action_size = 4
        self.min_action_torch = torch.tensor([-1, -1, -1, -1])
        self.max_action_torch = torch.tensor([1, 1, 1, 1])

        self.discount = TDMPC2["DISCOUNT"]
        self.horizon = TDMPC2["HORIZON"]
        self.mmpi_iterations = TDMPC2["MMPI_ITERATIONS"]
        self.num_trajectories = TDMPC2["NUM_TRAJECTORIES"]
        self.num_samples = TDMPC2["NUM_SAMPLES"]
        self.num_elites = TDMPC2["NUM_ELITES"]
        self.min_std = TDMPC2["MIN_STD"]
        self.max_std = TDMPC2["MAX_STD"]
        self.temperature = TDMPC2["TEMPERATURE"]
        self.latent_size = TDMPC2["LATENT_SIZE"]
        self.log_std_min = TDMPC2["LOG_STD_MIN"]
        self.log_std_max = TDMPC2["LOG_STD_MAX"]
        self.log_std_dif = self.log_std_max - self.log_std_min
        self.entropy_coef = TDMPC2["ENTROPY_COEF"]
        self.enc_lr_scale = TDMPC2["ENC_LR_SCALE"]
        self.consistency_coef = TDMPC2["CONSISTENCY_COEF"]
        self.reward_coef = TDMPC2["REWARD_COEF"]
        self.q_coef = TDMPC2["Q_COEF"]

        # MPPI: we save the mean from the previous planning in this variable to make the planning better
        self._prior_mean = torch.zeros(self.horizon - 1, self.action_size, device=self.device)

        self.encoder_net = EncoderNet(state_size=self.state_size, latent_size=self.latent_size).to(self.device)
        self.encoder_net.eval()
        self.dynamics_net = DynamicsNet(latent_size=self.latent_size, action_size=self.action_size).to(self.device)
        self.dynamics_net.eval()
        self.reward_net = RewardNet(latent_size=self.latent_size, action_size=self.action_size).to(self.device)
        self.reward_net.eval()

        self.policy_net = ActorNet(latent_size=self.latent_size, action_size=self.action_size).to(self.device)
        self.policy_net.eval()
        self.q1_net = CriticNet(latent_size=self.latent_size, action_size=self.action_size).to(self.device)
        self.q1_net.eval()
        self.q2_net = CriticNet(latent_size=self.latent_size, action_size=self.action_size).to(self.device)
        self.q2_net.eval()

        # Target nets
        self.q1_target_net = CriticNet(latent_size=self.latent_size, action_size=self.action_size).to(self.device)
        self.q2_target_net = CriticNet(latent_size=self.latent_size, action_size=self.action_size).to(self.device)
        self.q1_target_net.load_state_dict(self.q1_net.state_dict())
        self.q1_target_net.eval()
        self.q2_target_net.load_state_dict(self.q2_net.state_dict())
        self.q2_target_net.eval()

        if eval:
            self.loadModel(checkpoint_name)

    def __repr__(self):
        """
        For printing purposes only
        """
        return f"TDMPC2Agent"

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> np.ndarray:
        """
        Select an action by planning some steps in the future and take the best estimated action.

        During training, we add noise to the proposed action.
        """
        proposed_action = self._plan(state)

        return proposed_action

    def reset(self):
        self._prior_mean = torch.zeros(self.horizon - 1, self.action_size, device=self.device)

    def setMode(self, eval=False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        self.isEval = eval
        if self.isEval:
            self.encoder_net.eval()
            self.policy_net.eval()
            self.q1_net.eval()
            self.q2_net.eval()
        else:
            self.encoder_net.train()
            self.policy_net.train()
            self.q1_net.train()
            self.q2_net.train()

    def import_checkpoint(self, checkpoint: dict) -> None:
        self.encoder_net.load_state_dict(checkpoint["encoder_net"])
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.dynamics_net.load_state_dict(checkpoint["dynamics_net"])
        self.reward_net.load_state_dict(checkpoint["reward_net"])
        self.q1_net.load_state_dict(checkpoint["q1_net"])
        self.q2_net.load_state_dict(checkpoint["q2_net"])
        self.q1_target_net.load_state_dict(checkpoint["q1_target_net"])
        self.q2_target_net.load_state_dict(checkpoint["q2_target_net"])

    def export_checkpoint(self) -> dict:
        checkpoint = {
            "encoder_net": self.encoder_net.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "dynamics_net": self.dynamics_net.state_dict(),
            "reward_net": self.reward_net.state_dict(),
            "q1_net": self.q1_net.state_dict(),
            "q2_net": self.q2_net.state_dict(),
            "q1_target_net": self.q1_target_net.state_dict(),
            "q2_target_net": self.q2_target_net.state_dict(),
        }
        return checkpoint

    # ----------------------- Model Functions -----------------------

    @torch.no_grad()
    def _plan(self, state: torch.Tensor) -> np.ndarray:
        """
        Plan proposed_action sequence of action_sequence_samples using the learned world model.

        Args:
            state (torch.Tensor): Real state from which to plan in the future.

        Returns:
            torch.Tensor: Action to take in the environment at the current timestep.
        """
        # Step 01: we use our prior policy net to give some action proposals
        latent_state = self.encoder_net(state)
        pi_actions = torch.empty(self.horizon, self.num_trajectories, self.action_size, device=self.device)
        _latent_state = latent_state.repeat(self.num_trajectories, 1)
        for t in range(self.horizon - 1):
            pi_actions[t], _ = self._predict_action(_latent_state)
            _latent_state = self.dynamics_net(_latent_state, pi_actions[t])
        pi_actions[-1], _ = self._predict_action(_latent_state)

        # Step 02: The rest of the action proposals are done by the MPPI algorithm.
        # Initialize state and parameters to prepare MPPI.
        latent_state = latent_state.repeat(self.num_samples, 1)
        mean = torch.zeros(self.horizon, self.action_size, device=self.device)
        mean[:-1] = self._prior_mean  # get the mean from the planning before
        std = torch.full((self.horizon, self.action_size), self.max_std, dtype=torch.float, device=self.device)
        actions = torch.empty(self.horizon, self.num_samples, self.action_size, device=self.device)
        actions[:, :self.num_trajectories] = pi_actions

        # Iterate MPPI
        for _ in range(self.mmpi_iterations):
            # Step 03: Sample random actions by using the mean and std.
            r = torch.randn(self.horizon, self.num_samples - self.num_trajectories, self.action_size,
                            device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(self.min_action_torch, self.max_action_torch)
            actions[:, self.num_trajectories:] = actions_sample

            # Compute elite actions
            value = self._estimate_q_of_action_sequence(latent_state, actions)
            elite_idxs = torch.topk(value.squeeze(1), self.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update score, mean and std. parameters for the next iteration
            score = torch.softmax(self.temperature * elite_value, dim=0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (
                    score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.min_std, self.max_std)

        # Select action by sampling the one with the highest score w.r.t. gumbel noise perturbation
        gumbel_noise = torch.distributions.Gumbel(0, 1).sample(score.shape).to(self.device)
        noisy_scores = (score + gumbel_noise).squeeze()
        selected_index = torch.argmax(noisy_scores, dim=0)
        planned_action_sequence = elite_actions[:, selected_index, :]
        planned_action, std = planned_action_sequence[0], std[0]

        # save the mean for the next planning
        self._prior_mean = copy.deepcopy(mean[1:])
        return planned_action.clamp(self.min_action_torch, self.max_action_torch).cpu().numpy()

    # TODO
    def _predict_action(self, latent_state: torch.Tensor):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """

        # Step 01: Get the gaussian policy prior from the policy network
        mean, log_std = self.policy_net(latent_state).chunk(2, dim=-1)
        log_std = log_std_func(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        log_prob = gaussian_logprob_func(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1]
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = squash_func(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = {
            "mean": mean,
            "log_std": log_std,
            "action_prob": 1.,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def _min_q_value(self, state: torch.Tensor, action: torch.Tensor, use_target: bool):
        """
        Computes the minimum q value of a state-action pair.
        """
        if use_target:
            next_q1 = self.q1_target_net(state, action)
            next_q2 = self.q2_target_net(state, action)
        else:
            next_q1 = self.q1_net(state, action)
            next_q2 = self.q2_net(state, action)
        min_q_value = torch.min(next_q1, next_q2)
        return min_q_value

    @torch.no_grad()
    def _estimate_q_of_action_sequence(self, latent_state: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of a trajectory starting at latent state and executing a given sequence of actions.
        We base this calculation by a monte carlo estimate on a horizon-long trajectory.
        latent_space (latent_state)
        action_sequence (num_samples, horizon, action_size)
        """

        # Step 01: Let's first predict the state and the discounted reward in the num of `horizon` in the future.
        _G, _discount = 0, 1
        # latent_state = latent_state.unsqueeze(0).repeat(action_sequence.shape[0], 1)  # expand the latent space
        for action in action_sequence.unbind(0):
            reward = self.reward_net(latent_state, action)
            latent_state = self.dynamics_net(latent_state, action)
            _G += _discount * reward
            _discount *= self.discount

        # Step 02: Sample an action based on our policy
        action, _ = self._predict_action(latent_state)

        # Step 03: Finally, we compute the q value
        min_q_value = self._min_q_value(latent_state, action, use_target=False)
        final_q_value = _G + _discount * min_q_value
        return final_q_value

    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        try:
            checkpoint = torch.load(file_name, map_location=self.device)
            self.import_checkpoint(checkpoint)
            logging.info(f"Model for {self.__repr__()} loaded successfully from {file_name}")
        except FileNotFoundError:
            logging.error(f"Error: File {file_name} not found.")
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {str(e)}")