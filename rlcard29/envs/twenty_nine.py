"""
File: rlcard29/envs/twenty_nine.py
Author: Arnob Das
Date: 2025-06-28
"""

    
import numpy as np
from rlcard.envs import Env
from rlcard29.games.twenty_nine.game import TwentyNineGame
from rlcard29.games.twenty_nine.utils import encode_card, decode_card

class TwentyNineEnv(Env):
    def __init__(self, config=None):
        if config is None:
            config = {}
        if 'allow_step_back' not in config:
            config['allow_step_back'] = False
        if 'seed' not in config:
            config['seed'] = None
        self.name = 'twenty_nine'
        self.game = TwentyNineGame()
        super().__init__(config)
        self.action_num = self.game.get_num_actions()
        self.state_shape = [[128]] * self.game.num_players
        self.action_shape = [None for _ in range(self.game.num_players)]

    def _extract_state(self, state):
        obs = np.zeros(128, dtype=int)
        if 'hand' not in state or state['hand'] is None:
            print(f"Warning: Invalid state['hand']: {state}")
            state['hand'] = []
        hand_cards = np.zeros(32)
        for card in state['hand']:
            try:
                hand_cards[encode_card(card)] = 1
            except Exception as e:
                print(f"Error encoding card {card}: {e}")
        obs[0:32] = hand_cards
        raw_legal_actions = state.get('legal_actions', [])
        legal_actions = self._get_legal_actions_id(raw_legal_actions)
        action_mask = np.zeros(self.action_num, dtype=int)
        for action_id in legal_actions.keys():
            action_mask[action_id] = 1
        obs[32:32+self.action_num] = action_mask
        obs[0] = state.get('bid_value', 0)  # Fixed indexing bug
        return {
            'obs': obs,
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': raw_legal_actions  # Ensure consistency
        }                                                                       

    def _get_legal_actions_id(self, legal_actions):
        legal_actions_ids = {}
        for action in legal_actions:
            legal_actions_ids[self._encode_action(action)] = action
        return legal_actions_ids

    def _decode_action(self, action_id):
        if 32 <= action_id < 47:
            if action_id == 46:
                return 'pass'
            return str(action_id - 16)
        elif 47 <= action_id < 51:
            return {47: 'S', 48: 'H', 49: 'D', 50: 'C'}[action_id]
        else:
            return decode_card(action_id)

    def _encode_action(self, action):
        if isinstance(action, int):
            return action
        if action == 'pass':
            return 46
        elif action in ['S', 'H', 'D', 'C']:
            return {'S': 47, 'H': 48, 'D': 49, 'C': 50}[action]
        elif isinstance(action, str) and action.isdigit():
            return int(action) + 16
        else:
            return encode_card(action)

    def _get_payoffs(self):
        return self.game.get_payoffs()

    def get_payoffs(self):
        return self.game.get_payoffs()

    def get_detailed_result(self):
        return {
            'summary': self.game.get_payoffs(),
            'log': self.game.get_game_log(),
        }

    def get_perfect_information(self):
        return {}

    def run(self, is_training=False):
        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()
        while not self.is_over():
            extracted_state = self._extract_state(state)
            if is_training:
                action, info = self.agents[player_id].step(extracted_state)
            else:
                action = self.agents[player_id].eval_step(extracted_state)
            trajectories[player_id].append([state, action])
            next_state, next_player_id = self.step(action)
            state, player_id = next_state, next_player_id
        payoffs = self.get_payoffs()
        for player_id in range(self.num_players):
            trajectories[player_id].append(payoffs[player_id])
        return trajectories, payoffs