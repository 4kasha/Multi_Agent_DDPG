"""
    main code that contains the neural network setup
    policy + critic updates
    see ddpg.py for other details in the network

"""

import numpy as np
import torch
from .ddpg import DDPGAgent
from .utils import ReplayBuffer_MARL

class MultiAgents:
    def __init__(self, 
                 num_agents,
                 obs_size,
                 action_size, 
                 config,
                 seed,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        
        single_agent = DDPGAgent(
                 num_agents, 
                 obs_size, 
                 action_size, 
                 seed = seed, 
                 lr_actor = config['LR_ACTOR'],
                 lr_critic = config['LR_CRITIC'],
                 weight_decay = config['WEIGHT_DECAY'],
                 theta = config['THETA'],
                 sigma = config['SIGMA'],
                 tau = config['TAU'],
                 hidden_layers = config['hidden_layers'],
                 use_bn = config['use_bn'], 
                 use_reset = config['use_reset'], 
                 noise = config['noise'],
                 device = device)

        # multi agents set up
        self.agents = [single_agent for i in range(num_agents)]
        
        self.num_agents = num_agents
        self.obs_size = obs_size
        self.action_size = action_size
        self.config = config
        self.device = device

        self.batch_size = config['BATCH_SIZE']
        self.gamma = config['GAMMA']
        self.tau = config['TAU']
        self.update_every = config['UPDATE_EVERY']

        # Replay memory
        self.memory = ReplayBuffer_MARL(config['BUFFER_SIZE'], self.batch_size, seed, device)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, obs, actions, rewards, next_obs, dones):
        """Save experience in replay memory, and use random sample from buffer to learn.
        Shape
        ======
            obs (np.array):         (num_agents, obs_size)
            actions (np.array):     (num_agents, action_size)
            rewards (list):         (num_agents,)
            next_obs (np.array):    (num_agents, obs_size)
            dones (list,bool):      (num_agents,)
        """
        # Save experience / reward  --->  utils.py
        self.memory.add(obs, actions, rewards, next_obs, dones)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, obs, eps=0.):
        """Returns actions for given observations as per current policy.
        
        Params
        ======
            obs (np.array): current obs
            eps (float): epsilon, for exploration action space
            return (np.array): actions for each agent following their policy
        """
        actions = [agent.act(obs[i], eps) for i, agent in enumerate(self.agents)]
        actions = np.vstack(actions)

        return actions
    
    def collect_actions_from(self, obs, target):
        """Returns minibatch of actions for given observations as per current policy.
        
        Params
        ======
            obs (tensor): current obs from all agents(they might be non-contiguous), (batch_size, num_agents, obs_size)
            target (bool): actions from target or local policy network
            return (tensor): batch of actions for each agent following their policy, (batch_size, num_agents, action_size)
        """
        actions = [agent.get_tensor_act(obs[:,i].contiguous(), target) for i, agent in enumerate(self.agents)]
        actions = torch.stack(actions).transpose(1,0)

        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples from all agents.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (x, a, r, x', dones) tuples 
            gamma (float): discount factor
        Shape
        ======
            x : (batch_size, num_agents, obs_size)
            a : (batch_size, num_agents, action_size)
            r : (batch_size, num_agents)
            x': (batch_size, num_agents, obs_size)
            dones : (batch_size, num_agents)
        """
        obs, actions, rewards, next_obs, dones = experiences

        # reshape the size like (batch_size, num_agents*obs_size) for input
        obs_full = obs.reshape(self.batch_size,-1)
        next_obs_full = next_obs.reshape(self.batch_size,-1)

        # reshape the size like (batch_size, num_agents*action_size) for input
        actions_full = actions.reshape(self.batch_size,-1)
        next_actions_full = self.collect_actions_from(next_obs,target=True).reshape(self.batch_size,-1)  # for critic
        actions_pred_full = self.collect_actions_from(obs,target=False).reshape(self.batch_size,-1)      # for actor

        for i, agent in enumerate(self.agents):
            # update critics
            agent.update_critic(obs_full, actions_full, next_obs_full, next_actions_full, \
                                rewards[:,i].unsqueeze(1), dones[:,i].unsqueeze(1), gamma)

            # update actors
            agent.update_actor(obs_full, actions_pred_full)

            # update targets
            agent.update_targets()

