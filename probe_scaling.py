import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import subprocess
import sys
import glob

def run_test(num_agents, max_food, episodes=10):
    model_name = "model_N1_F10_S50_EF0.5"
    max_steps = 1000

    print(f"\n=== Testing N={num_agents}, F={max_food} ===")

    cmd = [
        sys.executable,
        "test_long.py",
        "--num_agents", str(num_agents),
        "--max_food", str(max_food),
        "--max_steps", str(max_steps),
        "--model", model_name,
        "--episodes", str(episodes)
    ]

    subprocess.run(cmd, capture_output=True, text=True)
    return True

def compute_avg_distance(positions):
    if len(positions) < 2:
        return 0.0
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = abs(positions[i] - positions[j])
            dist = min(dist, 50 - dist)
            distances.append(dist)
    return np.mean(distances)

def load_positions_for_n(n):
    pattern = f"raw_data/final_positions_N{n}_ep*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    all_positions = []
    for f in files:
        positions = np.loadtxt(f, dtype=int)
        all_positions.append(positions)
    return np.concatenate(all_positions)

def load_threshold_data(n):
    """Load threshold positions and avg_distance for a given N."""
    pattern = f"raw_data/threshold_N{n}_ep*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None
    all_positions = []
    avg_distances = []
    for f in files:
        with open(f, 'r') as file:
            lines = file.readlines()
            if len(lines) < 3:
                continue
            avg_dist = float(lines[1].split(',')[1])
            avg_distances.append(avg_dist)
            positions = [int(line.strip()) for line in lines[2:] if line.strip()]
            all_positions.extend(positions)
    if not all_positions:
        return None, None
    return np.array(all_positions), np.mean(avg_distances)

def plot_avg_distance_vs_n(agent_counts, avg_distances):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(agent_counts, avg_distances, 'o-', markersize=8, linewidth=2, color='blue')
    ax.set_xlabel('Number of Agents (N)')
    ax.set_ylabel('Average Inter-Agent Distance')
    ax.set_title('Scaling of Spatial Organization with Population Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/probe/avg_distance_vs_N.png', dpi=150)
    plt.close()

def plot_final_distributions(agent_counts):
    fig, ax = plt.subplots(figsize=(12, 8))
    for n in agent_counts:
        positions = load_positions_for_n(n)
        if positions is None:
            continue
        bins = np.arange(0, 51)
        counts, _ = np.histogram(positions, bins=bins)
        ax.plot(bins[:-1], counts, label=f'N={n}', linewidth=2)
    ax.set_xlabel('Position on Grid')
    ax.set_ylabel('Number of Agents')
    ax.set_title('Final Distribution of Agents Across Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/probe/final_distributions.png', dpi=150)
    plt.close()

def plot_threshold_distributions(agent_counts):
    fig, ax = plt.subplots(figsize=(12, 8))
    for n in agent_counts:
        positions, _ = load_threshold_data(n)
        if positions is None:
            continue
        bins = np.arange(0, 51)
        counts, _ = np.histogram(positions, bins=bins)
        ax.plot(bins[:-1], counts, label=f'N={n}', linewidth=2)
    ax.set_xlabel('Position on Grid')
    ax.set_ylabel('Number of Agents')
    ax.set_title('Agent Distribution at Food Threshold (Total Food < max_food)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/probe/threshold_distributions.png', dpi=150)
    plt.close()

def plot_threshold_avg_distance(agent_counts):
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_distances = []
    valid_agents = []
    for n in agent_counts:
        _, avg_dist = load_threshold_data(n)
        if avg_dist is None:
            continue
        avg_distances.append(avg_dist)
        valid_agents.append(n)
    if valid_agents:
        ax.plot(valid_agents, avg_distances, 'o-', markersize=8, linewidth=2, color='blue')
        expected_random = 50 / 3
        ax.axhline(y=expected_random, color='red', linestyle='--', 
                   label=f'Random expectation ({expected_random:.1f})')
        ax.set_xlabel('Number of Agents (N)')
        ax.set_ylabel('Avg Distance at Threshold')
        ax.set_title('Inter-Agent Distance at Food Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/probe/threshold_avg_distance.png', dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, nargs='+', 
                        default=[10, 20, 30, 40, 50],
                        help='List of agent counts to test')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of test episodes per N')
    args = parser.parse_args()

    os.makedirs('raw_data', exist_ok=True)
    os.makedirs('plots/probe', exist_ok=True)

    for n in args.agents:
        run_test(n, n, args.episodes)

    avg_distances = []
    valid_agents = []
    for n in args.agents:
        positions = load_positions_for_n(n)
        if positions is None:
            continue
        num_episodes = args.episodes
        if len(positions) % num_episodes != 0:
            num_episodes = len(positions) // (n if n > 0 else 1)
        episode_positions = positions.reshape(num_episodes, -1)
        episode_distances = [compute_avg_distance(ep) for ep in episode_positions]
        avg_dist = np.mean(episode_distances)
        avg_distances.append(avg_dist)
        valid_agents.append(n)

    if valid_agents:
        plot_avg_distance_vs_n(valid_agents, avg_distances)

    plot_final_distributions(args.agents)
    plot_threshold_distributions(args.agents)
    plot_threshold_avg_distance(args.agents)

if __name__ == "__main__":
    main()  