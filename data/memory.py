import numpy as np
from collections import namedtuple
from typing import Tuple

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)

class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = [None] * capacity  # Preallocate a fixed-size list
        self.size = 0
        self.pos = 0  # Pointer to the current insertion index

    def __len__(self) -> int:
        return self.size

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.
        
        If the buffer is full, the oldest experience is overwritten.
        """
        self.buffer[self.pos] = experience
        self.pos = (self.pos + 1) % self.capacity  # Circular increment
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> Tuple:
        # Randomly select indices from the valid portion of the buffer
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        # Retrieve experiences using constant-time list indexing
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, dones, next_states = zip(*experiences)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.float32),
            np.array(rewards,     dtype=np.float32),
            np.array(dones,       dtype=bool),
            np.array(next_states, dtype=np.float32),
        )