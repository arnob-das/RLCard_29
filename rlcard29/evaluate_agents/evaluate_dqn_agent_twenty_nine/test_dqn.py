import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.agents.dqn_agent import Transition
from rlcard.utils import tournament, get_device
import os
import torch
import numpy as np
import traceback
try:
    import rlcard29
except ImportError as e:
    print(f"Error importing rlcard29: {e}")
    raise

def main():
    save_dir = "rlcard29/models/dqn_model_twenty_nine"
    model_path = os.path.join(save_dir, 'checkpoint_dqn.pt')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run the training script first.")
        return

    try:
        device = get_device()
    except Exception as e:
        print(f"Error getting device: {e}")
        raise

    try:
        env = rlcard.make('twenty_nine')
    except Exception as e:
        print(f"Error creating twenty_nine environment: {e}")
        raise

    try:
        torch.serialization.add_safe_globals([
            Transition,
            np.ndarray,
            np._core.multiarray._reconstruct,
            np.dtypes.Int64DType,
        ])
    except AttributeError as e:
        print(f"Error setting safe globals: {e}")
        raise

    try:
        checkpoint = torch.load(model_path, weights_only=True, map_location=device)
        dqn_agent = DQNAgent.from_checkpoint(checkpoint=checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint with weights_only=True: {e}")
        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location=device)
            dqn_agent = DQNAgent.from_checkpoint(checkpoint=checkpoint)
        except Exception as e2:
            print(f"Failed to load checkpoint: {e2}")
            raise

    try:
        random_agent = RandomAgent(num_actions=env.num_actions)
        env.set_agents([dqn_agent, random_agent, random_agent, random_agent])
    except Exception as e:
        print(f"Error setting agents: {e}")
        raise

    print("Starting tournament: DQN agent vs. three Random agents...")
    try:
        rewards = tournament(env, 1000)
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
        print("Stack trace:")
        traceback.print_exc()
        print("Falling back to custom tournament loop...")
        try:
            payoffs = [0] * env.num_players
            num_games = 1000
            for game in range(num_games):
                env.reset()
                while not env.is_over():
                    player_id = env.get_player_id()
                    state = env.get_state(player_id)
                    action = env.agents[player_id].step(state)
                    env.step(action)
                _p = env.get_payoffs()
                for i in range(len(payoffs)):
                    payoffs[i] += _p[i]
            rewards = [p / num_games for p in payoffs]
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
        except Exception as e2:
            print(f"Custom tournament failed: {e2}")
            print("Stack trace:")
            traceback.print_exc()
            raise

if __name__ == '__main__':
    main()