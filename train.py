"""
Collaboration and Competition : Tennis env in Unity ML-Agents Environments

Multi Agent Deep Deterministic Policy Gradients (MADDPG) algorithm implementation.
cf. https://arxiv.org/pdf/1706.02275.pdf
"""

import numpy as np
from collections import deque
import pickle
import torch
from maddpg.agents import MultiAgents
from unityagents import UnityEnvironment

"""
Params
======
    n_episodes (int): maximum number of training episodes
    eps_start (float): starting value of epsilon, for exploration action space
    eps_end (float): minimum value of epsilon
    eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    goal_score (float): average score to be required
    env_file_name (str): your path to Tennis.app 
    config (dict): parameter set for training
"""

n_episodes=2000
eps_start=1.0
eps_end=0.01
eps_decay=0.9999
goal_score=0.5

cp_actor="cp_actor_from_agent_"
cp_critic="cp_critic_from_agent_"
scores_file="scores_maddpg.txt"

env_file_name="Tennis_Windows_x86_64/Tennis.exe"

config = {
    'BUFFER_SIZE': int(1e6),         # replay buffer size
    'BATCH_SIZE' : 256,              # minibatch size
    'GAMMA' : 0.99,                  # discount factor
    'TAU' :1e-3,                     # for soft update of target parameters
    'LR_ACTOR' : 1e-3,               # learning rate of the actor
    'LR_CRITIC' : 1e-3,              # learning rate of the critic
    'WEIGHT_DECAY' : 0,              # L2 weight decay
    'UPDATE_EVERY' : 1,              # how often to update the network
    'THETA' : 0.15,                  # parameter for Ornstein-Uhlenbeck process
    'SIGMA' : 0.2,                   # parameter for Ornstein-Uhlenbeck process and Gaussian noise
    'hidden_layers' : [256,128],     # size of hidden_layers
    'use_bn' : True,                 # use batch norm or not 
    'use_reset' : True,              # weights initialization used in original ddpg paper
    'noise' : "gauss"                # choose noise type, gauss(Gaussian) or OU(Ornstein-Uhlenbeck process) 
}

########  Environment Setting  ########
env = UnityEnvironment(file_name=env_file_name)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
#######################################


###########  Multi Agent Setting  ##########
multiagent = MultiAgents(num_agents, state_size, action_size, config, seed=0)
print('-------- Model structure --------')
print('-------- Actor --------')
print(multiagent.agents[0].actor_local)
print('-------- Critic -------')
print(multiagent.agents[0].critic_local)
print('---------------------------------')   
############################################

scores_agent = []                                          # list containing scores from each episode and agent
scores_window = deque(maxlen=100)                          # last 100 scores
eps = eps_start                                            # initialize epsilon
best_score = -np.inf
is_First = True

print('Interacting with env ...')   
for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
    states = env_info.vector_observations                  # get the current state                             
    multiagent.reset()
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = multiagent.act(states, eps)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        multiagent.step(states, actions, rewards, next_states, dones)
        states = next_states                               # roll over states to next time step
        scores += rewards                                  # update the score (for each agent)
        if np.any(dones):                                  # exit loop if episode finished
            break
    score = np.max(scores)
    scores_window.append(score)         # save most recent score
    scores_agent.append(score)          # save most recent score
    eps = max(eps_end, eps_decay*eps)   # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_window)), end="")
        
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window)>=goal_score and np.mean(scores_window)>=best_score:
        if is_First:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode-100, np.mean(scores_window)))
            is_First = False

        for i in range(num_agents):
                torch.save(multiagent.agents[i].actor_local.state_dict(), cp_actor + "{}.pth".format(i))
                torch.save(multiagent.agents[i].critic_local.state_dict(), cp_critic + "{}.pth".format(i))
        best_score = np.mean(scores_window)
        

f = open(scores_file, 'wb')
pickle.dump(scores_agent, f)

# End