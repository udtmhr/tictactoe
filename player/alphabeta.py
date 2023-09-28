import numpy as np
from base_player import Player

class AlphaBetaPlayer(Player):    
    def alphabeta(self, alpha, beta):
        if self.env.is_win():
            return 1
        elif self.env.is_lose():
            return -1
        elif self.env.is_drow():
            return 0
        
        for action in self.env.legal_action():
            self.env.put(action)
            self.env.change_turn()
            alpha = max(-self.alphabeta(-beta, -alpha), alpha)
            self.env.change_turn()
            self.env.pb[action] = 0

            if alpha > beta:
                return alpha
        return alpha
    
    def take_action(self):
        best_action = None
        alpha = -float("inf")
        for action in self.env.legal_action():
            self.env.put(action)
            self.env.change_turn()
            score = -self.alphabeta(-float("inf"), -alpha)
            self.env.change_turn()
            self.env.pb[action] = 0
            if score > alpha:
                alpha = score
                best_action = action
        return best_action
    