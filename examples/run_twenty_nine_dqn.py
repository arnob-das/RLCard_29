import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import reorganize, get_device
import os
import sys
from datetime import datetime
import torch
import numpy as np
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

if __name__ == '__main__':
    log_dir = "logs/run_twenty_nine_dqn"
    save_dir = "pretrained_agents/dqn"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"run_twenty_nine_dqn_log_{timestamp}.log"
    sys.stdout = Logger(log_dir, log_file)

    # Check for available device
    device = get_device()
    print(f"Using device: {device}")

    env = rlcard.make('twenty_nine')
    eval_env = rlcard.make('twenty_nine')

    # DQN agent parameters
    hidden_layers = [128,128,128,128,128]

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
    eval_every = 20
    eval_num = 100
    
    for episode in range(num_episodes):
        state, player_id = env.reset()
        
        while True:
            action = env.agents[player_id].step(state)
            next_state, next_player_id = env.step(action)
            done = env.is_over()
            
            # Reorganize the trajectory for the DQN agent
            if player_id == 0 :# If it was the DQN agent's turn
                reward = env.get_payoffs()[player_id]
                transition = (state, action, reward, next_state, done)
                dqn_agent.feed(transition)

            state = next_state
            player_id = next_player_id

            if done:
                break
        
        if (episode + 1) % eval_every == 0:
            print(f"\nEpisode {episode+1}/{num_episodes} - Evaluating...")
            # Evaluation logic can be added here if needed
            
    print("\nTraining complete.")

    # Save the trained model
    os.makedirs(save_dir, exist_ok=True)
    dqn_agent.save_checkpoint(save_dir)
    print(f"Model saved to {save_dir}") 