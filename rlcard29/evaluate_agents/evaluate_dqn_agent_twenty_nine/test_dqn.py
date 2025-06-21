import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.agents.dqn_agent import Transition
from rlcard.utils import tournament, get_device
import os
import torch
import numpy as np
try:
    import rlcard29
except ImportError as e:
    print(f"Error importing rlcard29: {e}")
    raise

def main():
    # Paths
    save_dir = "rlcard29/models/dqn_model_twenty_nine"
    model_path = os.path.join(save_dir, 'checkpoint_dqn.pt')

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run the training script first.")
        return

    # Device
    try:
        device = get_device()
        print(f"Running on device: {device}")
    except Exception as e:
        print(f"Error getting device: {e}")
        raise

    # Environment
    try:
        env = rlcard.make('twenty_nine')
        print("Twenty-Nine environment created successfully.")
    except Exception as e:
        print(f"Error creating twenty_nine environment: {e}")
        raise

    # Allowlist globals for PyTorch 2.7.1
    try:
        torch.serialization.add_safe_globals([
            Transition,
            np.ndarray,
            np._core.multiarray._reconstruct,
            np.dtype,  # Added for NumPy dtype
        ])
        print("Safe globals allowlisted successfully.")
    except AttributeError as e:
        print(f"Error setting safe globals: {e}")
        raise

    # Load DQN agent
    try:
        checkpoint = torch.load(model_path, weights_only=True, map_location=device)
        print("Checkpoint keys:", list(checkpoint.keys()))
        dqn_agent = DQNAgent.from_checkpoint(checkpoint=checkpoint)
        print("DQN agent loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint with weights_only=True: {e}")
        # Fallback: Load with weights_only=False
        try:
            print("Attempting to load with weights_only=False...")
            checkpoint = torch.load(model_path, weights_only=False, map_location=device)
            print("Checkpoint loaded with weights_only=False. Keys:", list(checkpoint.keys()))
            dqn_agent = DQNAgent.from_checkpoint(checkpoint=checkpoint)
            print("DQN agent loaded successfully with weights_only=False.")
        except Exception as e2:
            print(f"Failed to load with weights_only=False: {e2}")
            raise

    # Setup agents
    try:
        random_agent = RandomAgent(num_actions=env.num_actions)
        env.set_agents([dqn_agent, random_agent, random_agent, random_agent])
        print("Agents set successfully.")
    except Exception as e:
        print(f"Error setting agents: {e}")
        raise

    # Evaluate
    print("Starting tournament: DQN agent vs. three Random agents...")
    try:
        rewards = tournament(env, 1000)
        # Workaround for get_payoffs() returning int
        if isinstance(rewards, int):
            print("Warning: tournament returned int, converting to list")
            rewards = [rewards if i == 0 else 0 for i in range(env.num_players)]
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
    except Exception as e:
        print(f"Error during tournament: {e}")
        raise

if __name__ == '__main__':
    main()