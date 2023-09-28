import numpy as np

class RandomPlayer:
    def __init__(self, env):
        self.env = env
        
    def take_action(self):
        return np.random.choice(self.env.legal_action(), 1)[0]

class RandomPlayer2(RandomPlayer):
    def win_action(self):
        win_actions = []
        for action in self.env.legal_action():
            self.env.put(action)
            if self.env.is_win():
                win_actions.append(action)
            self.env.pb[action] = 0
        return win_actions
    
    def take_action(self):
        if (win_actions := self.win_action()):
            return np.random.choice(win_actions, 1)[0]
        else:
            return super().take_action()

class RandomPlayer3(RandomPlayer):
    def win_lose_action(self):
        win_action = []
        lose_action = []
        for action in self.env.legal_action():
            self.env.put(action)
            if self.env.is_win():
                win_action.append(action)
            self.env.pb[action] = 0
            if not win_action:
                self.env.change_turn()
                self.env.put(action)
                if self.env.is_win():
                    lose_action.append(action)
                self.env.pb[action] = 0
                self.env.change_turn()
        return win_action, lose_action
    
    def take_action(self):
        win_action, lose_action = self.win_lose_action()
        if win_action:
            return np.random.choice(win_action, 1)[0]
        elif lose_action:
            return np.random.choice(lose_action, 1)[0]
        return super().take_action()