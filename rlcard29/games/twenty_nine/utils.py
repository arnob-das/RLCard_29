# Utility functions for the 29 card game

SUITS = ['S', 'H', 'D', 'C']  # Spades, Hearts, Diamonds, Clubs
RANKS = ['J', '9', 'A', '10', 'K', 'Q', '8', '7']  # High to low
CARD_POINTS = {'J': 3, '9': 2, 'A': 1, '10': 1, 'K': 0, 'Q': 0, '8': 0, '7': 0}

# Card encoding: 0-31, where 0 = 'SJ', 1 = 'S9', ..., 31 = 'C7'
def encode_card(card):
    """Encode a card (e.g., 'SJ') to an integer 0-31."""
    suit, rank = card[0], card[1:]
    return SUITS.index(suit) * 8 + RANKS.index(rank)

def decode_card(idx):
    """Decode an integer 0-31 to a card string (e.g., 'SJ')."""
    suit = SUITS[idx // 8]
    rank = RANKS[idx % 8]
    return suit + rank

def get_card_points(card):
    """Return the point value of a card (e.g., 'SJ' -> 3)."""
    rank = card[1:]
    return CARD_POINTS[rank]

def get_deck():
    """Return a list of all 32 cards as strings."""
    return [s + r for s in SUITS for r in RANKS] 