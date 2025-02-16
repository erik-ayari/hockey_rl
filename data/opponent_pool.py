import random
import torch
import trueskill

from hockey.hockey_env import BasicOpponent
from model.modules import Actor

class OpponentPool():
    def __init__(
        self,
        pool_size: int,
        actor_params: dict
    ):
        self.basic_weak     = BasicOpponent(weak=True)
        self.basic_strong   = BasicOpponent(weak=False)
        self.snapshots      = []
        self.pool_size      = pool_size
        self.actor_params   = actor_params

        self.ratings = []
        self.ratings.append(trueskill.Rating())
        self.ratings.append(trueskill.Rating())

    def add_snapshot(self, actor):
        snapshot = Actor(
            state_dim   = self.actor_params["state_dim"],
            action_dim  = self.actor_params["action_dim"],
            num_layers  = self.actor_params["num_layers"],
            hidden_dim  = self.actor_params["hidden_dim"]
        )
        snapshot.load_state_dict(actor.state_dict())
        snapshot.eval()

        self.snapshots.append(snapshot)
        self.ratings.append(trueskill.Rating())

        print("[INFO] Added a new snapshot to the pool.")

        if len(self.snapshots) > self.pool_size:
            self.snapshots.pop(0)
            self.ratings.pop(2)
    
    def sample_opponent(self):
        self.opponent_idx = random.randint(0, len(self.snapshots) + 1)

    def set_opponent(self, idx):
        self.opponent_idx = idx

    def act_opponent(self, observation):
        if self.opponent_idx == 0:
            return self.basic_weak.act(observation)
        elif self.opponent_idx == 1:
            return self.basic_strong.act(observation)
        else:
            with torch.no_grad():
                observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
                return self.snapshots[self.opponent_idx - 2].forward(observation, deterministic=True)[0].numpy()

    def udpate_rating(self, model_rating, opponent_idx, outcome):
        if outcome == 1:
            model_rating, self.ratings[opponent_idx] = trueskill.rate_1vs1(model_rating, self.ratings[opponent_idx])
        elif outcome == -1:
            self.ratings[opponent_idx], model_rating = trueskill.rate_1vs1(self.ratings[opponent_idx], model_rating)
        else:
            model_rating, self.ratings[opponent_idx] = trueskill.rate_1vs1(model_rating, self.ratings[opponent_idx], drawn=True)
        return model_rating
    
    def get_rating(self, opponent_idx, sigma=False):
        if sigma:
            return self.ratings[opponent_idx].sigma
        return self.ratings[opponent_idx].mu

    def __len__(self):
        return len(self.snapshots) + 2