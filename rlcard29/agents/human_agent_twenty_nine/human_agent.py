"""
File: rlcard29/agents/human_agent_twenty_nine/human_agent.py
Author: Arnob Das
Date: 2025-06-28
"""

import random

def _print_state(state):
    """Prints the state of the game for a human player."""
    print("\n" + "="*30)
    print(f"   PLAYER {state['player_id']}'s TURN")
    print("="*30)
    print(f"   Match Score: Team 0 (P0,P2): {state['match_scores'][0]} | Team 1 (P1,P3): {state['match_scores'][1]}")
    print(f"   Phase: {state['phase']}")
    if state['phase'] != 'bidding':
        print(f"   Trump Suit: {state['trump_suit'] if state['trump_suit'] else 'Not Revealed'}")
    if state['bid_winner'] is not None:
        print(f"   Current Bid: {state['bid_value']} (Winner: P{state['bid_winner']})")
    if state['bid_history']:
        print(f"   Bid History: {state['bid_history']}")
    if state['trick']:
        print(f"   Current Trick: {[f'P{p}:{c}' for p, c in state['trick']]}")

    print(f"\n   Your Hand: {state['hand']}")
    print(f"\n   Legal Actions: {state['legal_actions']}")

class HumanAgent:
    """
    Human agent for the 29 card game. Interacts via console input.
    """
    def __init__(self, num_actions):
        self.use_raw = True
        self.num_actions = num_actions

    def step(self, state):
        _print_state(state['raw_obs'])
        action = input('Enter your action: ')
        while action not in state['raw_legal_actions']:
            print('!! Invalid action. Please choose from the legal actions. !!')
            action = input('Enter your action: ')
        
        # Find the ID for the chosen raw action
        for action_id, raw_action in state['legal_actions'].items():
            if raw_action == action:
                return action_id
        
        return None # Should not happen

    def eval_step(self, state):
        return self.step(state), {} 