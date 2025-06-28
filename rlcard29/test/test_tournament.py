"""
File: rlcard29/test/test_tournament.py
Author: Arnob Das
Date: 2025-06-28
"""
    
import rlcard
import rlcard29
from rlcard.agents import RandomAgent
from rlcard.utils import tournament
import traceback

env = rlcard.make('twenty_nine')
env.set_agents([RandomAgent(env.num_actions) for _ in range(4)])
try:
    rewards = tournament(env, 10)
    print(f"Tournament rewards: {rewards}, type: {type(rewards)}")
except Exception as e:
    print(f"Tournament failed: {e}")
    print("Stack trace:")
    traceback.print_exc()
    print("Falling back to custom tournament loop...")
    try:
        payoffs = [0] * env.num_players
        num_games = 10
        for game in range(num_games):
            print(f"Starting game {game+1}/{num_games}")
            env.reset()
            print(f"Game {game+1}: Reset complete")
            while not env.is_over():
                player_id = env.get_player_id()
                state = env.get_state(player_id)
                print(f"Game {game+1}: Player {player_id}, State: {state}")
                action = env.agents[player_id].step(state)
                print(f"Game {game+1}: Player {player_id}, Action: {action}")
                env.step(action)
                print(f"Game {game+1}: Step complete")
            _p = env.get_payoffs()
            print(f"DEBUG: Game {game+1}/{num_games}, get_payoffs() returned {_p}, type: {type(_p)}")
            for i in range(len(payoffs)):
                payoffs[i] += _p[i]
        rewards = [p / num_games for p in payoffs]
        print(f"Tournament rewards: {rewards}, type: {type(rewards)}")
    except Exception as e2:
        print(f"Custom tournament failed: {e2}")
        print("Stack trace:")
        traceback.print_exc()