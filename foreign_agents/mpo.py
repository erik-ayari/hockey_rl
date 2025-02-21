from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from torch import nn

"""
Compact discrete MPO agent that can be used for hockey inference

Author: Andre Pfrommer
"""

MPO = {
    "DISC_TO_CONT_TRAFO": True, 
    # If you want the disc actions to be transformed to cont actions via the discrete_to_continous_action func
    "HIDDEM_DIM": 256,
    "CHECKPOINT_NAME": None,
}

class Actor(nn.Module):
    """
    Policy network for continuous action space that outputs the mean and covariance matrix of 
    a multivariate Gaussian distribution
    
    - ds the dimension of the state space
    - da the dimension of the action space
    If action space discrete:
    - Softmax over all possible discrete actions da
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.ds = state_dim
        self.da = action_dim 
            
        #Feedforward network
        self.net = nn.Sequential(
            nn.Linear(self.ds, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        #Output layer here softmax over all possible discrete actions da
        self.lin3 = nn.Linear(hidden_dim, self.da)
        self.out = nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forwards input through the network
         First preprocessing the input state
         If action space discrete:
         - Softmax over all possible discrete actions da
         
        :param state: (B, ds), where B the batch size and ds the dimension of the state space
        :return: mean vector (B, da) and cholesky factorization of covariance matrix (B, da, da)
        """
        # Batch size
        B = state.size(0) 
        
        # Preprocess the input state
        x = self.net(state).to(self.device)  # (B, 256)
        # Output layer here softmax over all possible discrete actions da
        logits = self.out(self.lin3(x))
        return logits, None
    
    def greedyAction(self, state: torch.Tensor) -> torch.Tensor:
        """
        :state: (B, ds) the current state
        :return: (B,) the greedy action
        """
        with torch.no_grad():
            action_probs, _ = self.forward(state)
            greedyAction = torch.argmax(action_probs, dim = -1)
            return greedyAction


class MPODestilledAgent(nn.Module):
    def __init__(self, checkpoint_name: str, device: torch.device = torch.device("cpu"), eval=True):
        nn.Module.__init__(self)
        
        self.device = device
        self.ds = 18 # State dim in hockey env
        self.da = 7 # We need 7 discrete actions in hockey env
        self.keep_mode = True # From the hockey env
        self.hidden_dim = MPO["HIDDEM_DIM"]
        self.discrete_to_cont_trafo = MPO["DISC_TO_CONT_TRAFO"]
        
        # Actor net init 
        self.actor = Actor(state_dim=self.ds, action_dim=self.da, hidden_dim=self.hidden_dim, device=self.device).to(self.device)
        
        self.loadModel(checkpoint_name)
                           
    def discrete_to_continous_action(self, discrete_action):
        ''' 
        Copied from hockey_env.py
        Converts discrete actions into continuous ones (for each player)
        The actions allow only one operation each timestep, e.g. X or Y or angle change.
        This is surely limiting. Other discrete actions are possible
        Action 0: do nothing
        Action 1: -1 in x
        Action 2: 1 in x
        Action 3: -1 in y
        Action 4: 1 in y
        Action 5: -1 in angle
        Action 6: 1 in angle
        Action 7: shoot (if keep_mode is on)
        '''
        action_cont = [(discrete_action == 1) * -1.0 + (discrete_action == 2) * 1.0,  # player x
                    (discrete_action == 3) * -1.0 + (discrete_action == 4) * 1.0,  # player y
                    (discrete_action == 5) * -1.0 + (discrete_action == 6) * 1.0]  # player angle
        
        if self.keep_mode:
            action_cont.append((discrete_action == 7) * 1.0)

        return action_cont
        
    def act(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            # If you are in eval mode, get the greedy Action
            action = self.actor.greedyAction(state).item()  
            # Transform discrete to cont action
            if self.discrete_to_cont_trafo:
                action = self.discrete_to_continous_action(action)
        return action   
        
    def import_checkpoint(self, checkpoint: dict) -> None:
        # Here we only import the actor net
        self.actor.load_state_dict(checkpoint["actor"])

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
