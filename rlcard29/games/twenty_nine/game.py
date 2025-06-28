"""
File: rlcard29/games/twenty_nine/game.py
Author: Arnob Das
Date: 2025-06-28
"""
    
import random
from rlcard29.games.twenty_nine.dealer import Dealer
from rlcard29.games.twenty_nine.player import Player
from rlcard29.games.twenty_nine.judger import Judger
from rlcard29.games.twenty_nine.utils import get_deck

class TwentyNineGame:
    """
    Main game engine for the 29 card game (Bangladeshi variant).
    Handles state, actions, transitions, and scoring.
    """
    def __init__(self, allow_step_back=False):
        """Initialize the game."""
        self.allow_step_back = allow_step_back
        self.num_players = 4
        # Actions: 32 cards + 14 bids (16-29) + pass + 4 trump suits
        self.num_actions = 32 + 14 + 1 + 4
        self.players = [Player(i) for i in range(self.num_players)]
        self.dealer = Dealer()
        self.dealer_id = 3 # Start with Player 3 as dealer
        self.match_scores = [0, 0]
        self.game_points_for_round = [0, 0]
        self.bid_history = []
        self.bid_winner = None
        self.bid_value = 15
        self.trump_suit = None
        self.trump_revealed = False
        self.phase = 'deal'
        self.current_player = 0
        self.trick = []
        self.trick_leader = 0
        self.trick_history = []
        self.played_cards = set()
        self.winner = None
        self.logs = []  # Store detailed logs

    def log(self, message):
        self.logs.append(message)

    def init_game(self):
        """Initializes a new round, rotating the dealer."""
        self.dealer_id = (self.dealer_id + 1) % self.num_players
        self.log(f"--- New Round Started --- Dealer is Player {self.dealer_id} ---")
        self.log(f"Current Match Score: Team 0 (0,2): {self.match_scores[0]}, Team 1 (1,3): {self.match_scores[1]}")

        for p in self.players:
            p.reset()
        
        self.dealer.shuffle()
        self.dealer.deal(self.players, 4)
        
        self.phase = 'bidding'
        self.bid_history = []
        self.bid_value = 15
        self.bid_winner = None
        self.trump_suit = None
        self.trump_revealed = False
        self.trick = []
        self.trick_history = []
        
        self.current_player = (self.dealer_id + 1) % self.num_players
        self.trick_leader = self.current_player
        
        return self.get_state(self.current_player), self.current_player

    def step(self, action):
        """Processes an action based on the current game phase."""
        self.log(f"Player {self.current_player} attempts action: {action}")
        
        if self.phase == 'bidding':
            return self._step_bidding(action)
        elif self.phase == 'trump_selection':
            return self._step_trump_selection(action)
        elif self.phase == 'play':
            return self._step_play(action)
        return self.get_state(self.current_player), self.current_player

    def _step_bidding(self, action):
        if action == 'pass':
            self.bid_history.append((self.current_player, 'pass'))
        else: # Is a bid
            bid_val = int(action)
            self.bid_value = bid_val
            self.bid_winner = self.current_player
            self.bid_history.append((self.current_player, bid_val))

        # Check for end of bidding
        if len(self.bid_history) == 4 and self.bid_winner is None:
            self.log("All players passed. Redealing for a new round.")
            return self.init_game()
            
        if len(self.bid_history) >= 4:
            last_three_actions = [a[1] for a in self.bid_history[-3:]]
            if last_three_actions == ['pass', 'pass', 'pass'] and self.bid_winner is not None:
                self.phase = 'trump_selection'
                self.current_player = self.bid_winner
                self.log(f"Bidding finished. Player {self.bid_winner} wins with a bid of {self.bid_value}.")
                return self.get_state(self.current_player), self.current_player

        self.current_player = (self.current_player + 1) % self.num_players
        return self.get_state(self.current_player), self.current_player

    def _step_trump_selection(self, action):
        self.trump_suit = action
        self.log(f"Player {self.bid_winner} chose {action} as the trump suit (secretly).")
        self.dealer.deal(self.players, 4) # Deal remaining cards
        self.phase = 'play'
        self.current_player = self.trick_leader
        return self.get_state(self.current_player), self.current_player

    def _step_play(self, action):
        player_id = self.current_player
        player = self.players[player_id]

        # This is not an action, but a game event triggered by a player's inability to follow suit
        if not self.trump_revealed and len(self.trick) > 0:
            led_suit = self.trick[0][1][0]
            if not any(card[0] == led_suit for card in player.hand):
                self.trump_revealed = True
                self.log(f"Player {player_id} cannot follow suit. Trump is revealed: {self.trump_suit}")

        # Normal card play
        card = action
        player.hand.remove(card)
        self.trick.append((player_id, card))

        if len(self.trick) == 4:
            winner_id = self._resolve_trick()
            self.log(f"Trick: {[f'P{p}:{c}' for p, c in self.trick]} -> Winner: P{winner_id}")
            self.players[winner_id].taken_tricks.append([c for _, c in self.trick])
            self.trick = []
            self.trick_leader = winner_id
            self.current_player = winner_id
        else:
            self.current_player = (self.current_player + 1) % self.num_players

        if all(len(p.hand) == 0 for p in self.players):
            self.phase = 'end'
            self._update_match_scores()

        return self.get_state(self.current_player), self.current_player

    def _update_match_scores(self):
        if self.bid_winner is None:
            self.log("Round ended before a bid was made. No score change.")
            return
            
        team_points = [0, 0]
        for i, p in enumerate(self.players):
            for trick in p.taken_tricks:
                for card in trick:
                    team_points[i % 2] += self._card_points(card)

        bidding_team_id = self.bid_winner % 2
        defending_team_id = 1 - bidding_team_id
        
        bid_target = self.bid_value
        bid_successful = team_points[bidding_team_id] >= bid_target
        game_point_value = 1

        if bid_successful:
            self.match_scores[bidding_team_id] += game_point_value
            self.log(f"Team {bidding_team_id} fulfilled their bid of {self.bid_value} (target {bid_target}) by scoring {team_points[bidding_team_id]}. They WIN {game_point_value} point(s).")
        else:
            self.match_scores[bidding_team_id] -= game_point_value
            self.log(f"Team {bidding_team_id} FAILED their bid of {self.bid_value} (target {bid_target}) by scoring {team_points[bidding_team_id]}. They LOSE {game_point_value} point(s).")

    def _resolve_trick(self):
        led_suit = self.trick[0][1][0]
        highest_card_info = self.trick[0]

        for (player_id, card) in self.trick[1:]:
            # Current highest is trump
            if self.trump_revealed and highest_card_info[1][0] == self.trump_suit:
                if card[0] == self.trump_suit and self._card_rank(card) < self._card_rank(highest_card_info[1]):
                    highest_card_info = (player_id, card)
            # Current highest is not trump, but new card is
            elif self.trump_revealed and card[0] == self.trump_suit:
                highest_card_info = (player_id, card)
            # Neither are trump, but same suit as led
            elif card[0] == led_suit and highest_card_info[1][0] != self.trump_suit:
                 if self._card_rank(card) < self._card_rank(highest_card_info[1]):
                    highest_card_info = (player_id, card)
        
        return highest_card_info[0]
    
    def _card_rank(self, card):
        order = ['J', '9', 'A', '10', 'K', 'Q', '8', '7']
        return order.index(card[1:])

    def _card_points(self, card):
        points = {'J': 3, '9': 2, 'A': 1, '10': 1, 'K': 0, 'Q': 0, '8': 0, '7': 0}
        return points[card[1:]]

    def get_state(self, player_id):
        player = self.players[player_id]
        state = {
            'player_id': player_id,
            'hand': sorted(player.hand),
            'legal_actions': self.get_legal_actions(),
            'phase': self.phase,
            'bid_value': self.bid_value,
            'bid_winner': self.bid_winner,
            'bid_history': self.bid_history,
            'trick': self.trick,
            'trump_suit': self.trump_suit if self.trump_revealed else None,
            'match_scores': self.match_scores,
        }
        return state

    def get_legal_actions(self):
        player_id = self.current_player
        player = self.players[player_id]

        if self.phase == 'bidding':
            actions = ['pass']
            min_bid = self.bid_value + 1
            actions.extend([str(b) for b in range(min_bid, 29 + 1)])
            return actions

        if self.phase == 'trump_selection':
            return ['S', 'H', 'D', 'C']

        if self.phase == 'play':
            actions = []
            if self.bid_winner is None: return [] # Safeguard
            # Card play logic
            if not self.trick:  # Player is leading the trick
                actions.extend(player.hand)
                return actions

            led_suit = self.trick[0][1][0]
            if any(card[0] == led_suit for card in player.hand): # Must follow suit
                actions.extend([c for c in player.hand if c[0] == led_suit])
            else: # Cannot follow suit
                if self.trump_revealed:
                    trumps_in_hand = [c for c in player.hand if c[0] == self.trump_suit]
                    if trumps_in_hand:
                        actions.extend(trumps_in_hand)
                    else:
                        actions.extend(player.hand)
                else:
                    actions.extend(player.hand)
            return list(set(actions))
        return []

    def is_over(self):
        return self.phase == 'end'
        
    def get_match_winner(self):
        if self.match_scores[0] >= 6: return 0
        if self.match_scores[1] >= 6: return 1
        if self.match_scores[0] <= -6: return 1 # Team 0 loses, so Team 1 wins
        if self.match_scores[1] <= -6: return 0 # Team 1 loses, so Team 0 wins
        return None

    def get_payoffs(self):
        if self.bid_winner is None: 
            return [0,0,0,0]
            
        bidding_team_id = self.bid_winner % 2
        payoffs = [0, 0, 0, 0]
        
        # Determine winning team based on the bid result
        team_points = [0, 0]
        for i, p in enumerate(self.players):
            for trick in p.taken_tricks:
                for card in trick:
                    team_points[i % 2] += self._card_points(card)
        
        bid_target = self.bid_value # Simplified for this context
        bid_successful = team_points[bidding_team_id] >= bid_target

        if bid_successful:
            for i in range(4): payoffs[i] = 1 if i % 2 == bidding_team_id else -1
        else:
            for i in range(4): payoffs[i] = -1 if i % 2 == bidding_team_id else 1
        
        return payoffs

    def get_round_summary(self):
        """
        Returns a dictionary with a summary of the completed round.
        """
        summary = {
            'bid_value': self.bid_value,
            'bid_winner': self.bid_winner,
            'bidding_team_id': None,
            'trump_suit': self.trump_suit,
            'bid_successful': False,
            'team_points': [0, 0],
        }

        if self.bid_winner is None:
            return summary # Not enough info for a full summary

        bidding_team_id = self.bid_winner % 2
        summary['bidding_team_id'] = bidding_team_id

        team_points = [0, 0]
        for i, p in enumerate(self.players):
            for trick in p.taken_tricks:
                for card in trick:
                    team_points[i % 2] += self._card_points(card)
        summary['team_points'] = team_points

        bid_target = self.bid_value
        summary['bid_successful'] = team_points[bidding_team_id] >= bid_target
        return summary

    def get_num_players(self):
        return self.num_players

    def get_num_actions(self):
        return self.num_actions

    def get_player_id(self):
        return self.current_player

    def get_game_log(self):
        return self.logs 