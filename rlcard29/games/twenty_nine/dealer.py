import random
from rlcard29.games.twenty_nine.utils import get_deck

class Dealer:
    """
    Handles dealing, bidding, and trump selection for the 29 card game.
    """
    def __init__(self):
        self.deck = []
        self.trump_suit = None
        self.bid_winner = None
        self.bid_value = None

    def shuffle(self):
        self.deck = get_deck()
        random.shuffle(self.deck)

    def deal(self, players, num_cards=4):
        """Deal num_cards to each player."""
        for player in players:
            cards = [self.deck.pop() for _ in range(num_cards)]
            player.receive_cards(cards)

    def conduct_bidding(self, players, min_bid=16, max_bid=28):
        """Conduct the bidding phase (Bangladeshi rules). Returns (winner, bid_value)."""
        # For now, random bidding for demo; replace with full logic later
        bids = [random.randint(min_bid, max_bid) for _ in players]
        winner = bids.index(max(bids))
        self.bid_winner = winner
        self.bid_value = max(bids)
        return winner, max(bids)

    def select_trump(self, bidder, trump_suit):
        """Set the trump suit as chosen by the bidder."""
        self.trump_suit = trump_suit 