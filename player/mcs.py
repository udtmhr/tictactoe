import numpy as np
import copy

class MCSPlayer:
    def __init__(self, env, num=10):
        self.env = env
        self.num = num
    
    def random_action(self):
        return np.random.choice(self.env.legal_action(), 1)[0]

    def playout(self):
        if self.env.is_win():
            return 1
        elif self.env.is_lose():
            return -1
        elif self.env.is_drow():
            return 0
        
        action = self.random_action()
        self.env.put(action)
        self.env.change_turn()
        score = -self.playout()
        self.env.change_turn()
        self.env.pb[action] = 0
        return score
    
    def take_action(self):
        legal_actions = self.env.legal_action()
        values = np.zeros(len(legal_actions))
        for i, action in enumerate(legal_actions):
            for _ in range(self.num):
                self.env.put(action)
                self.env.change_turn()
                values[i] += -self.playout()
                self.env.change_turn()
                self.env.pb[action] = 0
        return legal_actions[np.argmax(values)]
    

class MCTSPlayer(MCSPlayer):
    def __init__(self, env, expnad_num=10, sim_num=100):
        self.env = env
        self.expand_num = expnad_num
        self.sim_num = sim_num
        self.w = 0
        self.n = 0
        self.child_nodes = []
    
    def expand(self):
        legal_actions = self.env.legal_action()
        for action in legal_actions:
            self.env.put(action)
            self.env.change_turn()
            self.child_nodes.append(MCTSPlayer(copy.deepcopy(self.env)))
            self.env.change_turn()
            self.env.pb[action] = 0

    def calc_ucb1(self, w, n, t):
        return w / n + (2 * np.log1p(t - 1) / n) ** 0.5
    
    def max_child_node(self):
        t = 0
        for child in self.child_nodes:
            if child.n == 0:
                return child
            t += child.n
        
        best_child = None
        best_value = -float("inf")
        for child in self.child_nodes:
            value = self.calc_ucb1(child.w, child.n, t)
            if value > best_value:
                value = best_value
                best_child = child
        return best_child
        
    def evaluate(self):
        if self.env.is_done():
            value = 1
            if self.env.is_lose():
                value = -1
            else:
                value = 0
            self.w += value
            self.n += 1
            return value
        
        if self.child_nodes:
            value = -self.max_child_node().evaluate()
            self.w += value
            self.n += 1
            return value
        
        else:
            value = self.playout()
            self.w += value
            self.n += 1

            if self.n == self.expand_num:
                self.expand()
            return value
        
    def take_action(self):
        self.expand()

        for _ in range(self.sim_num):
            self.evaluate()
        
        legal_actions = self.env.legal_action()
        best_action = None
        best_n = 0
        for i, child in enumerate(self.child_nodes):
            if child.n > best_n:
                best_n = child.n
                best_action = legal_actions[i]
        self.__init__(self.env, self.expand_num, self.sim_num)
        return best_action
        