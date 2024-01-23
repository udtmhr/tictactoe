import numpy as np

class AlphaBetaPlayer:    
    def __init__(self, env):
        self.env = env
        
    def alphabeta(self, alpha, beta):
        if (value := self.env.is_done()) is not None:
            return value
        
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
    