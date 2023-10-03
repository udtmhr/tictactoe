import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from collections import deque


class QNetwork:
    def __init__(self, state_size, action_size, path=None):
        if path is None:
            self.model = Sequential()
            self.model.add(Dense(16, activation="relu", input_shape=(state_size,)))
            self.model.add(Dense(16, activation="relu"))
            self.model.add(Dense(16, activation="relu"))
            self.model.add(Dense(action_size, activation="linear"))

            self.model.compile(loss="huber_loss", optimizer="adam")
        else:
            self.model = load_model(path)


class Memory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)
    
    def add(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        id_lst = np.random.choice(len(self.memory), batch_size, replace=False)
        return np.array(self.memory)[id_lst]
    
    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self, env, path=None):
        self.env = env
        self.state_size = self.env.action_size * 2
        self.main_qn = QNetwork(self.state_size, self.env.action_size, path)
    
    def take_action(self):
        legal_action = self.env.legal_action()
        q_values = self.main_qn.model(tf.expand_dims(self.get_state(), axis=0)).numpy()[0]
        return legal_action[np.argmax(q_values[legal_action])]
    
    def learn_take_action(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.env.legal_action(), 1)[0]
        else:
            return np.argmax(self.main_qn.model(tf.expand_dims(self.get_state(), axis=0))[0])
    
    def get_state(self):
        return np.array([self.env.pb.copy(), self.env.ob.copy()]).reshape(-1)

    def make_data(self, batch_size, gamma):
        inputs = np.zeros((batch_size, self.state_size))
        outputs = np.zeros((batch_size, self.env.action_size))
        minibatch = self.memory.sample(batch_size)
        for i, (pre_state, pre_action, reward, state, done) in enumerate(minibatch):
            inputs[i] = pre_state
            if not done:
                reward += gamma * np.amax(self.target_qn.model(tf.expand_dims(state, axis=0))[0])
            outputs[i] = self.main_qn.model(tf.expand_dims(pre_state, axis=0))[0]
            outputs[i][pre_action] = reward
        return inputs, outputs

    def play(self, epsilon, batch_size, gamma, com):
        self.env.reset()
        player, opponent = np.random.choice((1, -1), 2, replace=False)
        state = self.get_state()
        pre_action = None
        while True:
            if self.env.turn == player:
                action = self.learn_take_action(epsilon)
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
            state = self.get_state()

            if (pre_action is not None and self.env.turn == opponent) or done:
                self.memory.add((pre_state, pre_action, reward, state, done))

            if len(self.memory) > batch_size:
                inputs, outputs = self.make_data(batch_size, gamma)
                self.main_qn.model.fit(inputs, outputs, epochs=1, verbose=0)

            if done:
                break
            self.env.change_turn()
    
    def learn(self, episode_num, e_start, e_end, e_decay_rate, gamma, com, path, memory_size=1000, batch_size=32):
        self.target_qn = QNetwork(self.state_size, self.env.action_size)
        self.memory = Memory(memory_size)
        for episode in range(episode_num):
            self.target_qn.model.set_weights(self.main_qn.model.get_weights())

            epsilon = e_end + (e_start - e_end) * np.exp(-e_decay_rate * episode)
            self.play(epsilon, batch_size, gamma, com)

            if episode % 1000 == 0:
                print(f"episode: {episode}, epsilon: {epsilon}")
        self.main_qn.model.save(path)

if __name__ == "__main__":
    import sys
    sys.path.append('/content/drive/MyDrive/Colab Notebooks/Tictactoe')
    import matplotlib.pyplot as plt
    from tictactoe import TicTacToe
    from random_player import RandomPlayer
    from dqn import DQN, Memory

    env = TicTacToe()
    dqn = DQN(env)
    com = RandomPlayer(env)
    path = r"C:\Users\tomhi\python\tictactoe\qnet\model1"
    dqn.learn(10000, 1, 0.01, 0.001, 0.95, com, path)
