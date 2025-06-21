import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import tournament, get_device
import rlcard29
import os
import torch

def main():
    # Directories and paths
    save_dir = "rlcard29/models/dqn_model_twenty_nine"
    model_path = os.path.join(save_dir, 'checkpoint_dqn.pt')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run the training script first.")
        return

    # Check for available device
    device = get_device()
    print(f"Using device: {device}")

    # Create the environment
    env = rlcard.make('twenty_nine')

    # Load the trained DQN agent
    dqn_agent = DQNAgent.from_checkpoint(checkpoint=torch.load(model_path))
    
    print("DQN agent loaded successfully.")

    # Create random agents for the opposing team
    random_agent = RandomAgent(num_actions=env.action_num)

    # Set up the agents in the environment to match the training configuration.
    # Player 0 will be the DQN agent.
    # Players 1, 2 & 3 will be Random agents.
    env.set_agents([dqn_agent, random_agent, random_agent, random_agent])

    print("Starting tournament: DQN agent vs. three Random agents...")
    rewards = tournament(env, 1000)
    
    # Print out the results
    print("\nTournament Results:")
    print("="*20)
    dqn_reward = rewards[0]
    random_rewards_avg = (rewards[1] + rewards[2] + rewards[3]) / 3

    print(f"DQN Agent (Player 0) average reward: {dqn_reward}")
    print(f"Random Agents (Players 1, 2, 3) average reward: {random_rewards_avg}")

    if dqn_reward > random_rewards_avg:
        print("\nDQN Agent wins!")
    else:
        print("\nRandom Agents win!")

if __name__ == '__main__':
    main() 