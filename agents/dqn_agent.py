import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor  # ADD THIS IMPORT
from environment.foraging_env import ForagingEnv
from environment.wrapper import SingleAgentWrapper
import os

class DQNAgent:
    """
    Wrapper for DQN agent using Stable-Baselines3 with shared policy.
    """
    
    def __init__(
        self,
        env: ForagingEnv,
        learning_rate: float = 1e-3,
        buffer_size: int = 50000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.3,
        exploration_final_eps: float = 0.02,
        verbose: int = 1
    ):
        self.env = env
        self.model = None
        self.params = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'learning_starts': learning_starts,
            'batch_size': batch_size,
            'tau': tau,
            'gamma': gamma,
            'train_freq': train_freq,
            'target_update_interval': target_update_interval,
            'exploration_fraction': exploration_fraction,
            'exploration_final_eps': exploration_final_eps,
            'verbose': verbose
        }
    
    def train(
        self,
        total_timesteps: int = 50000,
        eval_freq: int = 5000,
        eval_episodes: int = 5
    ) -> None:
        """
        Train the DQN agent.
        """
        # Create logs directory
        os.makedirs('./logs', exist_ok=True)
        
        # Create wrapped environment with Monitor for logging
        wrapped_env = SingleAgentWrapper(self.env)
        monitored_env = Monitor(wrapped_env, filename='./logs/monitor.csv')  # ADD MONITOR
        vec_env = DummyVecEnv([lambda: monitored_env])
        
        # Create evaluation environment
        eval_wrapped = SingleAgentWrapper(
            ForagingEnv(
                grid_size=self.env.grid_size,
                num_agents=self.env.num_agents,
                max_food=self.env.max_food,
                food_regrow_prob=self.env.food_regrow_prob,
                max_steps=self.env.max_steps
            )
        )
        eval_monitored = Monitor(eval_wrapped, filename='./logs/eval_monitor.csv')  # ADD MONITOR
        
        # Create model
        self.model = DQN(
            'MlpPolicy',
            vec_env,
            **self.params
        )
        
        # Set up evaluation callback
        eval_callback = EvalCallback(
            eval_monitored,
            best_model_save_path='./logs/best_model/',
            log_path='./logs/',
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save final model
        self.model.save('dqn_foraging_model')
    
    def load(self, path: str) -> None:
        """Load a pre-trained model."""
        self.model = DQN.load(path)
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Get action from the model."""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        action, _ = self.model.predict(observation, deterministic=False)
        return action
    
    def predict_batch(self, observations: np.ndarray) -> np.ndarray:
        """Get actions for all agents."""
        actions = []
        for obs in observations:
            action = self.predict(obs)
            actions.append(action)
        return np.array(actions)