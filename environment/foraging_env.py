import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

class ForagingEnv(gym.Env):
    """
    A 1D foraging environment with multiple agents.
    Agents move left/right, stay, or eat food at their current position.
    NO EXCLUDED VOLUME - agents can occupy the same cell.
    """
    
    def __init__(
        self,
        grid_size: int = 50,
        num_agents: int = 10,
        max_food: int = 10,
        food_regrow_prob: float = 0.0,
        max_steps: int = 50
    ):
        super(ForagingEnv, self).__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.max_food = max_food
        self.food_regrow_prob = food_regrow_prob
        self.max_steps = max_steps
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_food,
            shape=(1,),
            dtype=np.float32
        )
        
        self.food = None
        self.agent_positions = None
        self.step_count = 0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Initialize food grid
        self.food = np.random.randint(0, self.max_food + 1, size=self.grid_size)
        
        # Initialize agent positions randomly WITHOUT excluded volume
        # Agents can start on the same cell
        self.agent_positions = np.random.randint(0, self.grid_size, size=self.num_agents)
        
        self.step_count = 0
        
        observations = self._get_observations()
        info = {}
        return observations, info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        # Apply actions
        self._apply_actions(actions)
        
        # Regrow food (disabled by default)
        self._regrow_food()
        
        # Calculate rewards
        rewards = self._calculate_rewards(actions)
        
        self.step_count += 1
        
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        observations = self._get_observations()
        
        info = {}
        return observations, rewards, terminated, truncated, info
    
    def _apply_actions(self, actions: np.ndarray) -> None:
        """Apply actions to all agents. NO collision resolution."""
        new_positions = self.agent_positions.copy()
        
        for i, action in enumerate(actions):
            if action == 0:  # LEFT
                new_positions[i] = (self.agent_positions[i] - 1) % self.grid_size
            elif action == 1:  # RIGHT
                new_positions[i] = (self.agent_positions[i] + 1) % self.grid_size
            elif action == 2:  # STAY
                new_positions[i] = self.agent_positions[i]
            elif action == 3:  # EAT
                new_positions[i] = self.agent_positions[i]
                if self.food[self.agent_positions[i]] > 0:
                    self.food[self.agent_positions[i]] -= 1
        
        # NO collision resolution - agents can overlap
        self.agent_positions = new_positions
    
    def _regrow_food(self) -> None:
        """Randomly regrow food at each cell."""
        for i in range(self.grid_size):
            if np.random.random() < self.food_regrow_prob:
                self.food[i] = min(self.food[i] + 1, self.max_food)
    
    def _calculate_rewards(self, actions: np.ndarray) -> np.ndarray:
        """Calculate rewards for all agents."""
        rewards = np.zeros(self.num_agents)
        for i, action in enumerate(actions):
            if action == 3 and self.food[self.agent_positions[i]] > 0:
                rewards[i] = 1.0
        return rewards
    
    def _get_observations(self) -> np.ndarray:
        observations = np.zeros((self.num_agents, 1), dtype=np.float32)  # Only 1 feature
        for i, pos in enumerate(self.agent_positions):
            observations[i, 0] = self.food[pos]
        return observations