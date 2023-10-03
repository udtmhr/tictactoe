import numpy as np
import os

class QLPlayer:
    def __init__(self, env, path=None):
        self.env = env
        self.action_num = env.size ** 2
        self.state_num = 3 ** self.action_num
        if path is None:
            self.qtable = np.zeros((self.state_num, self.action_num))
        else:
            self.load_qtable(path)  
    
    def take_action(self):
        legal_actions = self.env.legal_action()
        max_id = np.argmax(self.qtable[self.env.get_state()][legal_actions])
        return legal_actions[max_id]
    
    def learn_take_action(self, epsilon, state):
        if np.random.rand() < epsilon:
            return np.random.choice(self.env.legal_action(), 1)[0]
        else:
            return np.argmax(self.qtable[state])
    
    def update_qtable(self, state, action, next_state, reward, done, alpha, gamma):
        max_q = 0 if done else np.argmax(self.qtable[next_state])
        self.qtable[state][action] += alpha * (reward + gamma * max_q - self.qtable[state][action])
        
    def save_qtable(self, path):
        np.save(path, self.qtable)
    
    def load_qtable(self, path):
        self.qtable = np.load(path)

    def learn(self, episode_num, alpha, gamma, epsilon, com, path):
        for i in range(episode_num + 1):
            if i % 1000 == 0:
                print(f"learning {i}episode")
            self.env.reset()
            player, opponent = np.random.choice((-1, 1), 2, replace=False)
            state = self.env.get_state()
            pre_action = None
            while True:
                if self.env.turn == player:
                    action = self.learn_take_action(epsilon, state)
                    if self.env.check_action(action):
                        self.env.put(action)
                        reward = self.env.reward()
                        done = self.env.is_done()
                    else:
                        reward = -1
                        done = True
                    pre_state = state
                    pre_action = action
                else:
                    action = com.take_action()
                    self.env.put(action)
                    reward = self.env.reward() * -1
                    done = self.env.is_done()
                state = self.env.get_state()

                if (pre_action is not None and self.env.turn == opponent) or done:
                    self.update_qtable(pre_state, pre_action, state, reward, done, alpha, gamma)
                if done:
                    break
                self.env.change_turn()
        self.save_qtable(path)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append("C:\\Users\\tomhi\\python\\tictactoe")
    from tictactoe import TicTacToe
    from random_player import RandomPlayer

    env = TicTacToe()
    ql = QLPlayer(env)
    ql.learn(10000, 0.1, 0.9, 0.1, RandomPlayer(env), r"player\qtable\table1")
