import os
import sys
from datetime import datetime
import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import get_device, tournament
import torch
import numpy as np
import traceback
try:
    import rlcard29
except ImportError as e:
    print(f"Error importing rlcard29: {e}")
    raise

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
    log_dir = "rlcard29/logs/train_log/dqn_model_train_log"
    save_dir = "rlcard29/models/dqn_model_twenty_nine"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"run_twenty_nine_dqn_log_{timestamp}.log"
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(log_dir, log_file)

    try:
        device = get_device()
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Error getting device: {e}")
        raise

    try:
        env = rlcard.make('twenty_nine')
        eval_env = rlcard.make('twenty_nine')
        print("Twenty-Nine environment created successfully.")
    except Exception as e:
        print(f"Error creating twenty_nine environment: {e}")
        raise

    hidden_layers = [128, 128, 128, 128, 128]
    state_shape = env.state_shape[0]
    try:
        dqn_agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=state_shape,
            mlp_layers=hidden_layers,
            device=device
        )
        random_agent = RandomAgent(num_actions=env.num_actions)
        env.set_agents([dqn_agent, random_agent, random_agent, random_agent])
        eval_env.set_agents([dqn_agent, random_agent, random_agent, random_agent])
        print("Agents initialized successfully.")
    except Exception as e:
        print(f"Error initializing agents: {e}")
        raise

    print_header("DQN Training on Twenty-Nine (5 layers, 100 episodes)")
    num_episodes = 100
    eval_every = 20
    eval_num = 100

    for episode in range(num_episodes):
        try:
            state, player_id = env.reset()
            while not env.is_over():
                action = env.agents[player_id].step(state)
                next_state, next_player_id = env.step(action)
                if player_id == 0:
                    payoffs = env.get_payoffs()
                    print(f"DEBUG: get_payoffs() returned {payoffs}, type: {type(payoffs)}")
                    if not isinstance(payoffs, (list, tuple, np.ndarray)):
                        print(f"Warning: get_payoffs() returned {type(payoffs)}: {payoffs}")
                        payoffs = [payoffs if i == 0 else 0 for i in range(env.num_players)]
                    reward = payoffs[player_id]
                    transition = (state, action, reward, next_state, env.is_over())
                    dqn_agent.feed(transition)
                state, player_id = next_state, next_player_id
            if (episode + 1) % eval_every == 0:
                print(f"\nEpisode {episode+1}/{num_episodes} - Evaluating...")
                try:
                    rewards = tournament(eval_env, eval_num)
                    print(f"DEBUG: tournament returned {rewards}, type: {type(rewards)}")
                    print(f"DQN reward: {rewards[0]}, Random avg: {(rewards[1] + rewards[2] + rewards[3]) / 3}")
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    print("Stack trace:")
                    traceback.print_exc()
                    print("Falling back to custom tournament loop...")
                    try:
                        payoffs = [0] * eval_env.num_players
                        for game in range(eval_num):
                            print(f"Evaluating game {game+1}/{eval_num}")
                            eval_env.reset()
                            print(f"Game {game+1}: Reset complete")
                            while not eval_env.is_over():
                                player_id = eval_env.get_player_id()
                                state = eval_env.get_state(player_id)
                                print(f"Game {game+1}: Player {player_id}, State: {state}")
                                action = eval_env.agents[player_id].step(state)
                                print(f"Game {game+1}: Player {player_id}, Action: {action}")
                                eval_env.step(action)
                                print(f"Game {game+1}: Step complete")
                            _p = eval_env.get_payoffs()
                            print(f"DEBUG: Game {game+1}/{eval_num}, get_payoffs() returned {_p}, type: {type(_p)}")
                            for i in range(len(payoffs)):
                                payoffs[i] += _p[i]
                        rewards = [p / eval_num for p in payoffs]
                        print(f"DEBUG: tournament returned {rewards}, type: {type(rewards)}")
                        print(f"DQN reward: {rewards[0]}, Random avg: {(rewards[1] + rewards[2] + rewards[3]) / 3}")
                    except Exception as e2:
                        print(f"Custom tournament failed: {e2}")
                        print("Stack trace:")
                        traceback.print_exc()
                        raise
        except Exception as e:
            print(f"Error in episode {episode+1}: {e}")
            print("Stack trace:")
            traceback.print_exc()
            raise

    print("\nTraining complete.")

    try:
        os.makedirs(save_dir, exist_ok=True)
        dqn_agent.save_checkpoint(save_dir, 'checkpoint_dqn.pt')
        print(f"Model saved to {save_dir}/checkpoint_dqn.pt")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        raise