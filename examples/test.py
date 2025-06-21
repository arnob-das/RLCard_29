import rlcard29

from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import tournament, get_device
import rlcard

import os
import torch

def main():
    # Directories and paths
    save_dir = "pretrained_agents/dqn"
    model_path = os.path.join(save_dir, 'checkpoint_dqn.pt')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run the training script first.")
        return

    # Check for available device
    device = get_device()
    print(f"Using device: {device}")

    # Create the environment
    env = rlcard.make('twenty_nine')

    # Safely load the trained DQN agent
    try:
        import rlcard.agents.dqn_agent  # Import necessary module
        with torch.serialization.safe_globals([rlcard.agents.dqn_agent.Transition]):
            checkpoint = torch.load(model_path)
            dqn_agent = DQNAgent.from_checkpoint(checkpoint=checkpoint)
        print("DQN agent loaded successfully.")
    except Exception as e:
        print("Error loading DQN agent:", e)
        return

    # Create random agents for the opposing team
    random_agent = RandomAgent(num_actions=env.action_num)

    # Set up the agents in the environment to match the training configuration
    # Player 0 will be the DQN agent
    env.set_agents([dqn_agent, random_agent, random_agent, random_agent])

    print("Starting tournament: DQN agent vs. three Random agents...")
    rewards = tournament(env, 1000)

    # Print out the results
    print("\nTournament Results:")
    print("=" * 20)
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
