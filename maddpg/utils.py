# utility function for implementing agents

import numpy as np
import random
from collections import namedtuple, deque
import torch

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer_MARL:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["observations", "actions", "rewards", "next_observations", "dones"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, obs, actions, rewards, next_obs, dones):
        """Add a new experience to memory.
        Shape
        ======
            obs (array):         (num_agents, obs_size)
            actions (array):     (num_agents, action_size)
            rewards (list):      (num_agents,)
            next_obs (array):    (num_agents, obs_size)
            dones (list,bool):   (num_agents,)
        """
        e = self.experience(obs, actions, rewards, next_obs, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # transpose and convert to numpy array
        obs, actions, rewards, next_obs, dones = map(lambda x: np.asarray(x), zip(*experiences))

        # convert to torch tensor
        obs = torch.from_numpy(obs).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)

        return (obs, actions, rewards, next_obs, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


