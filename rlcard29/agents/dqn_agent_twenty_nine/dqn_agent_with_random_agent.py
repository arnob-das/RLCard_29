import rlcard
from rlcard.agents import RandomAgent,DQNAgent
import rlcard29
import time
import os
import sys
from datetime import datetime
from rlcard.agents.dqn_agent import Transition
import numpy as np
from rlcard.utils import tournament, get_device
import torch


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
    save_dir = "rlcard29/models/dqn_model_twenty_nine"
    model_path = os.path.join(save_dir, 'checkpoint_dqn.pt')
    device = get_device()
    log_dir = "rlcard29/logs/play_log/dqn_agent_play_twenty_nine_log"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"run_twenty_nine_random_log_{timestamp}.log"
    sys.stdout = Logger(log_dir, log_file)

    env = rlcard.make('twenty_nine')
    try:
        torch.serialization.add_safe_globals([
            Transition,
            np.ndarray,
            np._core.multiarray._reconstruct,
            np.dtypes.Int64DType,
            np.dtype,
            np._core.multiarray.scalar,
            np.dtypes.Float64DType
        ])
    except AttributeError as e:
        print(f"Error setting safe globals: {e}")
        raise

    try:
        checkpoint = torch.load(model_path, weights_only=True, map_location=device)
        dqn_agent = DQNAgent.from_checkpoint(checkpoint=checkpoint)
        print("DQN agent loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint with weights_only=True: {e}")
        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location=device)
            dqn_agent = DQNAgent.from_checkpoint(checkpoint=checkpoint)
            print("DQN agent loaded with weights_only=False")
        except Exception as e2:
            print(f"Failed to load checkpoint: {e2}")
            raise
    env.set_agents([
        dqn_agent,
        RandomAgent(num_actions=env.action_num),
        RandomAgent(num_actions=env.action_num),
        RandomAgent(num_actions=env.action_num),
    ])

    print_header("29 Random Agent Match")

    while env.game.get_match_winner() is None:
        print_header(f"Starting New Round")
        state, player_id = env.reset()
            
        last_logs_len = 0
        round_over = False
        while not round_over:
            action, _ = env.agents[player_id].eval_step(state)
            state, player_id = env.step(action)
                
            # Print the logs for the move
            current_logs = env.game.get_game_log()
            new_logs = current_logs[last_logs_len:]
            if new_logs:
                print("\n--- Game Update ---")
                for log_item in new_logs:
                    print(f"  -> {log_item}")
            last_logs_len = len(current_logs)
            time.sleep(0.1) # slow down for readability

            if state['raw_obs']['phase'] == 'end':
                round_over = True

        # Round is over, print summary
        summary = env.game.get_round_summary()
        print("\n--- Round Summary ---")
        if summary['bid_winner'] is not None:
            print(f"  Bid: {summary['bid_value']} by Player {summary['bid_winner']} (Team {summary['bidding_team_id']})")
            print(f"  Trump: {summary['trump_suit']}")
            print(f"  Team {summary['bidding_team_id']} {'WON' if summary['bid_successful'] else 'LOST'} the bid.")
            print(f"  Final Points: Team 0: {summary['team_points'][0]} | Team 1: {summary['team_points'][1]}")
        else:
            print("  Round ended with no successful bid.")
        
        print(f"\n  Match Score: Team 0: {env.game.match_scores[0]} | Team 1: {env.game.match_scores[1]}")

    print_header("Match Over!")
    winner = env.game.get_match_winner()
    print(f"Team {winner} wins the match!")
    print(f"Final Score: Team 0: {env.game.match_scores[0]} | Team 1: {env.game.match_scores[1]}") 