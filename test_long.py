import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from environment.foraging_env import ForagingEnv
from agents.dqn_agent import DQNAgent

def gini_coefficient(values):
    if len(values) == 0:
        return None
    values = np.array(values)
    if np.sum(values) == 0:
        return 0.0
    n = len(values)
    sorted_values = np.sort(values)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_values)) - (n + 1) * np.sum(sorted_values)) / (n * np.sum(sorted_values))
    return gini

def compute_inter_agent_distance(positions, grid_size):
    n = len(positions)
    if n < 2:
        return 0.0
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            dist = abs(positions[i] - positions[j])
            dist = min(dist, grid_size - dist)
            distances.append(dist)
    return np.mean(distances)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=10)
    parser.add_argument('--max_food', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--grid_size', type=int, default=50)
    parser.add_argument('--model', type=str, default='model_N1_F10_S50_EF0.5')
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()
    
    os.makedirs('plots/testing', exist_ok=True)
    os.makedirs('raw_data', exist_ok=True)
    
    all_metrics = []
    
    for ep in range(args.episodes):
        print(f"\nEpisode {ep+1}/{args.episodes}")
        
        env = ForagingEnv(
            grid_size=args.grid_size,
            num_agents=args.num_agents,
            max_food=args.max_food,
            food_regrow_prob=0.0,
            max_steps=args.max_steps
        )
        
        agent = DQNAgent(env=env)
        try:
            agent.load(f"models/{args.model}.zip")
        except:
            print(f"Model models/{args.model}.zip not found.")
            return
        
        obs, info = env.reset()
        initial_food = env.food.copy()
        
        positions_history = []
        food_history = []
        rewards_history = []
        
        threshold_logged = False
        food_thr = args.max_food  # Threshold: when total food drops below food_thr
        
        for step in range(env.max_steps):
            actions = agent.predict_batch(obs)
            obs, rewards, terminated, truncated, info = env.step(actions)
            positions_history.append(env.agent_positions.copy())
            food_history.append(env.food.copy())
            rewards_history.append(np.sum(rewards))
            
            # Check threshold
            total_food = np.sum(env.food)
            if total_food < food_thr and not threshold_logged:
                threshold_logged = True
                threshold_positions = env.agent_positions.copy()
                threshold_step = step
                threshold_avg_distance = compute_inter_agent_distance(threshold_positions, args.grid_size)
                # Save to file
                with open(f'raw_data/threshold_N{args.num_agents}_ep{ep}.csv', 'w') as f:
                    f.write(f"step,{threshold_step}\n")
                    f.write(f"avg_distance,{threshold_avg_distance}\n")
                    for pos in threshold_positions:
                        f.write(f"{pos}\n")
                print(f"  Threshold logged at step {threshold_step}, total_food={total_food}")
            
            if terminated or truncated:
                break
        
        positions_history = np.array(positions_history)
        food_history = np.array(food_history)
        
        final_food = env.food.copy()
        final_positions = env.agent_positions.copy()
        
        total_food_consumed = np.sum(initial_food) - np.sum(final_food)
        
        steps = len(positions_history)
        distance_over_time = []
        gini_over_time = []
        for t in range(steps):
            positions = positions_history[t]
            dist = compute_inter_agent_distance(positions, args.grid_size)
            distance_over_time.append(dist)
            
            per_agent_food = np.zeros(args.num_agents)
            for step_idx in range(t+1):
                for i in range(args.num_agents):
                    pos = positions_history[step_idx, i]
                    if step_idx > 0:
                        food_eaten = food_history[step_idx-1][pos] - food_history[step_idx][pos]
                        if food_eaten > 0:
                            per_agent_food[i] += food_eaten
            gini = gini_coefficient(per_agent_food)
            gini_over_time.append(gini if gini is not None else 0.0)
        
        metrics = {
            'episode': ep,
            'total_reward': sum(rewards_history),
            'total_food_consumed': total_food_consumed,
            'initial_food': initial_food,
            'final_food': final_food,
            'positions_history': positions_history,
            'food_history': food_history,
            'rewards_history': rewards_history,
            'distance_over_time': distance_over_time,
            'gini_over_time': gini_over_time
        }
        all_metrics.append(metrics)
        
        print(f"  Reward: {metrics['total_reward']}, Food: {metrics['total_food_consumed']}")
    
    plot_metrics(all_metrics, args)

def plot_metrics(all_metrics, args):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    ax = axes[0, 0]
    positions = all_metrics[-1]['positions_history']
    for i in range(args.num_agents):
        ax.plot(positions[:, i], alpha=0.5, label=f'Agent {i}' if i < 5 else '')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position')
    ax.set_title(f'Agent Trajectories (N={args.num_agents})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(all_metrics[-1]['rewards_history'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Reward')
    ax.set_title('Rewards per Step')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    food_history = all_metrics[-1]['food_history']
    total_food = [np.sum(f) for f in food_history]
    ax.plot(total_food)
    ax.axhline(y=total_food[-1], color='r', linestyle='--', label=f'Final: {total_food[-1]:.0f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Food')
    ax.set_title('Total Food Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[1, 0]
    initial_food = all_metrics[-1]['initial_food']
    ax.bar(range(args.grid_size), initial_food, color='green', alpha=0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Food Amount')
    ax.set_title(f'Initial Food (Total: {np.sum(initial_food)})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    final_food = all_metrics[-1]['final_food']
    ax.bar(range(args.grid_size), final_food, color='red', alpha=0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Food Amount')
    ax.set_title(f'Final Food (Total: {np.sum(final_food)})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    initial_total = np.sum(all_metrics[-1]['initial_food'])
    final_total = np.sum(all_metrics[-1]['final_food'])
    ax.bar(['Initial', 'Final'], [initial_total, final_total], color=['green', 'red'])
    ax.set_ylabel('Total Food')
    ax.set_title(f'Food Depletion: {initial_total - final_total} consumed')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    steps = range(len(all_metrics[-1]['distance_over_time']))
    ax.plot(steps, all_metrics[-1]['distance_over_time'], color='blue')
    ax.axhline(y=np.mean(all_metrics[-1]['distance_over_time']), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_metrics[-1]["distance_over_time"]):.2f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Avg Distance')
    ax.set_title('Inter-Agent Distance Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[2, 1]
    gini_vals = all_metrics[-1]['gini_over_time']
    if gini_vals:
        ax.plot(steps, gini_vals, color='purple')
        ax.axhline(y=np.mean(gini_vals), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(gini_vals):.3f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Food Intake Inequality Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[2, 2]
    cum_reward = np.cumsum(all_metrics[-1]['rewards_history'])
    ax.plot(steps, cum_reward, color='orange')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/testing/test_metrics_N{args.num_agents}_F{args.max_food}_S{args.max_steps}.png', dpi=150)
    plt.close()
    print(f"\nMetrics plot saved to plots/testing/")
    
    print("\n" + "="*50)
    print(f"SUMMARY: {args.num_agents} agents, max_food={args.max_food}, max_steps={args.max_steps}")
    print("="*50)
    avg_reward = np.mean([m['total_reward'] for m in all_metrics])
    avg_consumed = np.mean([m['total_food_consumed'] for m in all_metrics])
    print(f"Average total reward: {avg_reward:.2f} +/- {np.std([m['total_reward'] for m in all_metrics]):.2f}")
    print(f"Average food consumed: {avg_consumed:.2f}")

if __name__ == "__main__":
    main()