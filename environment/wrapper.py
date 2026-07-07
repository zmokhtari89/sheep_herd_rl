import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SingleAgentWrapper(gym.Env):
    """
    Wrapper that makes a multi-agent environment look like a single-agent environment.
    For training: uses agent 0's observation and applies action to all agents.
    """
    
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__()
        self.env = env
        self.num_agents = env.num_agents
        
        # Each agent observes 5 features
        self.observation_space = spaces.Box(
            low=0,
            high=env.max_food + 1,
            shape=(5,),
            dtype=np.float32
        )
        
        # DQN requires Discrete action space
        self.action_space = spaces.Discrete(4)
        
        # Store observations for all agents
        self.agents_observations = None
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents_observations = obs
        # Return observation for first agent (shared policy applied to all)
        return obs[0], info
        
    def step(self, action):
        # Apply the same action to all agents
        actions = np.full(self.env.num_agents, action)
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.agents_observations = obs
        # Return observation for first agent and total reward
        return obs[0], np.sum(rewards), terminated, truncated, info