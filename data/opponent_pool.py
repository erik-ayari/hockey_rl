import random
import torch
import trueskill
import numpy as np
from itertools import accumulate

from hockey.hockey_env import BasicOpponent
from model.modules import Actor

class OpponentPool():
    def __init__(
        self,
        actor_params: dict,
        foreign_agents : dict = {},
        weighting : list = [],
        device = torch.device("cpu")
    ):
        self.device = device

        self.foreign_agents = foreign_agents
        self.foreign_agents_exist = (len(self.foreign_agents) > 0)

        self.actor_params = actor_params
    
        required_weightings = 3 if self.foreign_agents_exist else 2

        assert(len(weighting) == required_weightings)
        assert(sum(weighting) == 1.0)

        self.weighting = list(accumulate(weighting))

        self.ratings = {
            "basic": {
                "weak": trueskill.Rating(),
                "strong": trueskill.Rating()
            },
            "snapshots": []
        }

        self.model_rating = trueskill.Rating()

        if self.foreign_agents_exist:
            ratings_foreign = {}
            for name in self.foreign_agents.keys():
                ratings_foreign[name] = trueskill.Rating()
            self.ratings["foreign"] = ratings_foreign

        self.basic_weak     = BasicOpponent(weak=True)
        self.basic_strong   = BasicOpponent(weak=False)

        self.snapshots = []

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
        self.ratings["snapshots"].append(trueskill.Rating())

        print("[INFO] Added a new snapshot to the pool.")
    
    def sample_opponent(self):
        choice = random.random()
        choice_idx = 0
        for cum_prob in self.weighting:
            if choice > cum_prob:
                choice_idx += 1
            else:
                break
        if choice_idx == 0:
            self.opponent_type = "basic"
            self.opponent = "weak" if random.random() < 0.5 else "strong"
        if self.foreign_agents_exist:
            if choice_idx == 1:
                self.opponent_type = "foreign"
                self.opponent = random.choice(list(self.foreign_agents.keys()))
            if choice_idx == 2:
                if len(self.snapshots) == 0:
                    self.sample_opponent()
                    return
                self.opponent_type = "snapshots"
                self.opponent = random.randint(0, len(self.snapshots) - 1)

    def act_opponent(self, observation):
        if self.opponent_type == "basic":
            if self.opponent == "weak":
                return self.basic_weak.act(observation)
            elif self.opponent == "strong":
                return self.basic_strong.act(observation)
        else:
            with torch.no_grad():
                observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
                if self.opponent_type == "foreign":
                    return np.array(self.foreign_agents[self.opponent].act(observation))
                else:
                    return np.array(self.snapshots[self.opponent].act(observation))

    def udpate_rating(self, outcome):
        print(f"Updating rating with outcome {outcome} of {self.opponent_type}, {self.opponent}")
        if outcome == 1:
            self.model_rating, self.ratings[self.opponent_type][self.opponent] = trueskill.rate_1vs1(self.model_rating, self.ratings[self.opponent_type][self.opponent])
        elif outcome == -1:
            self.ratings[self.opponent_type][self.opponent], self.model_rating = trueskill.rate_1vs1(self.ratings[self.opponent_type][self.opponent], self.model_rating)
        else:
            self.model_rating, self.ratings[self.opponent_type][self.opponent] = trueskill.rate_1vs1(self.model_rating, self.ratings[self.opponent_type][self.opponent], drawn=True)
    
    def get_ratings(self):
        return self.ratings, self.model_rating