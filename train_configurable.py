import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from environment.foraging_env import ForagingEnv
from agents.dqn_agent import DQNAgent

def main():
    parser = argparse.ArgumentParser(description='Train DQN on foraging environment')
    parser.add_argument('--num_agents', type=int, default=1)
    parser.add_argument('--max_food', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--food_regrow', type=float, default=0.0)
    parser.add_argument('--total_timesteps', type=int, default=50000)
    parser.add_argument('--grid_size', type=int, default=50)
    parser.add_argument('--output', type=str, default='model')
    parser.add_argument('--exploration_fraction', type=float, default=0.3)
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots/training', exist_ok=True)
    
    # Create environment
    env = ForagingEnv(
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        max_food=args.max_food,
        food_regrow_prob=args.food_regrow,
        max_steps=args.max_steps
    )
    
    # Create DQN agent
    agent = DQNAgent(
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=0.02,
        verbose=1
    )
    
    # Train
    print(f"\nTraining: {args.num_agents} agents, max_food={args.max_food}, max_steps={args.max_steps}")
    print(f"Exploration fraction: {args.exploration_fraction}")
    
    agent.train(
        total_timesteps=args.total_timesteps,
        eval_freq=args.total_timesteps // 10,
        eval_episodes=5
    )
    
    # Save model in models/ folder
    model_name = f"models/{args.output}_N{args.num_agents}_F{args.max_food}_S{args.max_steps}_EF{args.exploration_fraction}.zip"
    agent.model.save(model_name)
    print(f"Model saved as {model_name}")
    
    # Plot training metrics (only first row)
    plot_training_metrics(args)

def plot_training_metrics(args):
    """Plot the rolling-average learning curve from saved logs."""
    log_dir = './logs/'
    monitor_file = os.path.join(log_dir, 'monitor.csv')

    if not os.path.exists(monitor_file):
        print("No monitor.csv found. Skipping training plots.")
        return

    data = pd.read_csv(monitor_file, skiprows=1)

    fig, ax = plt.subplots(figsize=(7, 5))

    reward_col = 'r' if 'r' in data.columns else ('l' if 'l' in data.columns else None)
    if reward_col and len(data[reward_col]) >= 10:
        rolling_avg = data[reward_col].rolling(window=10).mean()
        ax.plot(rolling_avg, color='orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (10-episode rolling average)')
        ax.set_title('Learning Curve')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough data for rolling avg', ha='center', va='center')
        ax.set_title('Learning Curve (not enough data)')

    plt.tight_layout()
    plt.savefig(f'plots/training/training_metrics_N{args.num_agents}_F{args.max_food}.png', dpi=150)
    plt.close()
    print("Training metrics saved to plots/training/")

if __name__ == "__main__":
    main()