import numpy as np
from rlcard.envs import Env
from rlcard29.games.twenty_nine.game import TwentyNineGame
from rlcard29.games.twenty_nine.utils import encode_card, decode_card

class TwentyNineEnv(Env):
    """
    Gym-like environment for the 29 card game, compatible with RLCard API.
    """
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
        self.state_shape = [[128]] * self.game.num_players  # Expanded state
        self.action_shape = [None for _ in range(self.game.num_players)]
        # TODO: Set up state/action spaces, etc.

    def _extract_state(self, state):
        obs = np.zeros(128, dtype=int)
        
        # Hand cards
        hand_cards = np.zeros(32)
        for card in state['hand']:
            hand_cards[encode_card(card)] = 1
        obs[0:32] = hand_cards
        
        # Legal actions mask
        legal_actions = self._get_legal_actions_id(state['legal_actions'])
        action_mask = np.zeros(self.action_num, dtype=int)
        for action_id in legal_actions.keys():
            action_mask[action_id] = 1
        obs[32:32+self.action_num] = action_mask

        # Other game info
        obs[32+self.action_num] = state['bid_value']
        
        return {
            'obs': obs,
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': state['legal_actions'],
        }

    def _get_legal_actions_id(self, legal_actions):
        legal_actions_ids = {}
        for action in legal_actions:
            legal_actions_ids[self._encode_action(action)] = action
        return legal_actions_ids

    def _decode_action(self, action_id):
        """Converts an action id to a raw action string."""
        if 32 <= action_id < 47: # Bid actions 16-29 -> 32-45, pass is 46
            if action_id == 46:
                return 'pass'
            return str(action_id - 16)
        elif 47 <= action_id < 51: # Trump suits
            return {47: 'S', 48: 'H', 49: 'D', 50: 'C'}[action_id]
        else: # Card play
            return decode_card(action_id)

    def _encode_action(self, action_str):
        """Converts a raw action string to an action id."""
        if action_str == 'pass':
            return 46
        elif action_str in ['S', 'H', 'D', 'C']:
            return {'S': 47, 'H': 48, 'D': 49, 'C': 50}[action_str]
        elif action_str.isdigit():
            return int(action_str) + 16
        else: # It's a card
            return encode_card(action_str)

    def _get_payoffs(self):
        return self.game.get_payoffs()

    def _get_done(self):
        return self.game.get_match_winner() is not None

    def get_perfect_information(self):
        """Return perfect information for the current state."""
        return {}

    def get_payoffs(self):
        """
        Returns a list of payoffs for each player.
        Payoffs are returned from the perspective of the bidding team.
        """
        return self.game.get_payoffs()

    def get_detailed_result(self):
        """Return a detailed summary and logs for the last game."""
        return {
            'summary': self.game.get_payoffs(),
            'log': self.game.get_game_log(),
        } 