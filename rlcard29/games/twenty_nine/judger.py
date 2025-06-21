from rlcard29.games.twenty_nine.utils import get_card_points

class Judger:
    """
    Handles scoring, win/loss, and special rules for the 29 card game.
    """
    def __init__(self, players):
        self.players = players
        # TODO: Initialize score tracking

    def calculate_points(self, tricks):
        """Calculate points for each team based on tricks won."""
        # Assume players 0/2 vs 1/3 are partners
        team_points = [0, 0]
        for i, player in enumerate(self.players):
            for trick in player.taken_tricks:
                for card in trick:
                    team_points[i % 2] += get_card_points(card)
        return team_points

    def check_win(self, bid, points):
        """Determine if the bidding team has won."""
        return points >= bid

    def calculate_adjusted_bid(self, bid, marriage_declared, doubled, redoubled):
        """
        Calculate the adjusted bid considering marriage, double, redouble, set.
        Returns (adjusted_bid, multiplier)
        """
        adjusted_bid = bid
        multiplier = 1
        if marriage_declared:
            if bid > 18:
                adjusted_bid = max(16, bid - 3)
        if doubled:
            multiplier *= 2
        if redoubled:
            multiplier *= 2
        return adjusted_bid, multiplier 