import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import reorganize, get_device
import os
import sys
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import rlcard29

class Logger:
    def __init__(self, log_dir, file_name):
        self.terminal = sys.stdout
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log = open(os.path.join(log_dir, file_name), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def print_header(message):
    print("\n" + "="*30)
    print(f"  {message}")
    print("="*30)

def plot_metrics(win_rates, losses, save_dir, timestamp, dqn_wins_total, random_wins_total):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot win rate
    plt.figure(figsize=(10, 5))
    plt.plot(win_rates, label='Win Rate (Team 0)')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Win Rate')
    plt.title('DQN Agent Win Rate Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'win_rate_{timestamp}.png'))
    plt.close()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('DQN Training Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'loss_curve_{timestamp}.png'))
    plt.close()

    # Plot bar chart for total wins
    plt.figure(figsize=(8, 5))
    agents = ['DQN Agent', 'Random Agents']
    wins = [dqn_wins_total, random_wins_total]
    plt.bar(agents, wins, color=['blue', 'orange'])
    plt.xlabel('Agent Type')
    plt.ylabel('Number of Wins')
    plt.title('Total Wins in 100 Episodes')
    plt.savefig(os.path.join(save_dir, f'wins_bar_chart_{timestamp}.png'))
    plt.close()

if __name__ == '__main__':
    log_dir = "rlcard29/logs/train_log/dqn_model_train_log"
    save_dir = "rlcard29/models/dqn_model_twenty_nine"
    plot_dir = "rlcard29/plots"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"run_twenty_nine_dqn_log_{timestamp}.log"
    sys.stdout = Logger(log_dir, log_file)

    # Check for available device
    device = get_device()
    print(f"Using device: {device}")

    env = rlcard.make('twenty_nine')
    eval_env = rlcard.make('twenty_nine')

    hidden_layers = [128, 128, 128, 128, 128]

    state_shape = env.state_shape[0]
    dqn_agent = DQNAgent(
        num_actions=env.action_num,
        state_shape=state_shape,
        mlp_layers=hidden_layers,
        device=device
    )
    random_agent = RandomAgent(num_actions=env.action_num)
    env.set_agents([dqn_agent, random_agent, random_agent, random_agent])
    eval_env.set_agents([dqn_agent, random_agent, random_agent, random_agent])

    print_header("DQN Training on 29 (5 layers, 100 episodes)")
    num_episodes = 100
    eval_every = 10
    eval_num = 100
    
    # Track metrics
    win_rates = []
    losses = []
    dqn_wins_total = 0
    random_wins_total = 0
    
    for episode in range(num_episodes):
        state, player_id = env.reset()
        
        while True:
            action = env.agents[player_id].step(state)
            next_state, next_player_id = env.step(action)
            done = env.is_over()
            
            # Reorganize the trajectory for the DQN agent
            if player_id == 0:  # If it was the DQN agent's turn
                reward = env.get_payoffs()[player_id]
                transition = (state, action, reward, next_state, done)
                loss = dqn_agent.feed(transition)
                if loss is not None and loss > 0:  # Ensure meaningful loss values
                    losses.append(float(loss))
            
            state = next_state
            player_id = next_player_id

            if done:
                break
        
        if (episode + 1) % eval_every == 0:
            print(f"\nEpisode {episode+1}/{num_episodes} - Evaluating...")
            # Evaluation
            wins = 0
            for _ in range(eval_num):
                state, player_id = eval_env.reset()
                while True:
                    action = eval_env.agents[player_id].step(state)
                    state, player_id = eval_env.step(action)
                    if eval_env.is_over():
                        payoffs = eval_env.get_payoffs()
                        if payoffs[0] > 0:  # DQN agent (team 0) wins
                            wins += 1
                        break
            win_rate = wins / eval_num
            win_rates.append(win_rate)
            print(f"Win Rate: {win_rate:.3f} ({wins}/{eval_num} games)")
            dqn_wins_total += wins
            random_wins_total += (eval_num - wins)  # Assuming random agents win the rest
    
    print("\nTraining complete.")
    
    # Save plots
    plot_metrics(win_rates, losses, plot_dir, timestamp, dqn_wins_total, random_wins_total)
    print(f"Plots saved to {plot_dir}")

    # Save the trained model
    os.makedirs(save_dir, exist_ok=True)
    dqn_agent.save_checkpoint(save_dir)
    print(f"Model saved to {save_dir}")