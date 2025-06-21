from rlcard29.games.twenty_nine.utils import encode_card, decode_card

class Player:
    """
    Represents a player in the 29 card game.
    Manages hand, actions, and special moves (marriage, double, redouble, set).
    """
    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = []  # List of card strings, e.g., ['SJ', 'H9', ...]
        self.taken_tricks = []  # List of lists of cards won in tricks
        # TODO: Add more player state as needed

    def receive_cards(self, cards):
        self.hand.extend(cards)

    def play_card(self, card):
        """Play a card from hand."""
        if card in self.hand:
            self.hand.remove(card)
            return card
        raise ValueError(f"Card {card} not in hand")

    def reset(self):
        self.hand = []
        self.taken_tricks = []

    def set(self):
        """Declare set if the bidding team fails their bid."""
        pass 