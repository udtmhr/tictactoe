import os
import pickle
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, Manager, cpu_count
from tensorflow.keras.layers import Dense, Conv2D, Input, Activation, Add, BatchNormalization, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback


class DualNet:
    def __init__(self, input_shape, output_shape, filter, residual_num, path):
        self.input_shape = input_shape
        self.make_model(path, output_shape, filter, residual_num)

    def conv(self, filter):
        return Conv2D(filter, 3, padding="same", use_bias=False, 
                    kernel_initializer="he_normal", kernel_regularizer=l2(0.0005))
        
    def residual_block(self, filter):
        def f(x):
            sc = x
            x = self.conv(filter)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = self.conv(filter)(x)
            x = BatchNormalization()(x)
            x = Add()([x, sc])
            x = Activation("relu")(x)
            return x
        return f

    def make_model(self, path, output_shape, filter, residual_num):
        if os.path.exists(os.path.join(path, "best")):
            self.model = load_model(os.path.join(path, "best"))
            return
        
        input = Input(shape=self.input_shape)

        x = self.conv(filter)(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        for _ in range(residual_num):
            x = self.residual_block(filter)(x)

        policy = Conv2D(2, 1, padding="same", kernel_initializer="he_normal", 
                        kernel_regularizer=l2(0.0005))(x)
        policy = BatchNormalization()(policy)
        policy = Activation("relu")(policy)
        policy = Flatten()(policy)
        policy = Dense(output_shape, activation="softmax", kernel_regularizer=l2(0.0005), name="policy")(policy)
        
        value = Conv2D(1, 1, padding="same", kernel_initializer="he_normal", 
                        kernel_regularizer=l2(0.0005))(x)
        value = BatchNormalization()(value)
        value = Activation("relu")(value)
        value = Flatten()(value)
        value = Dense(filter, activation="relu", kernel_regularizer=l2(0.0005))(value)
        value = Dense(1, activation="tanh", kernel_regularizer=l2(0.0005), name="value")(value)

        self.model = Model(inputs=input, outputs=[policy, value])

        k.clear_session()

    def predict(self, env):
        a, b, c = self.input_shape
        x = np.array([env.pb, env.ob])
        x = x.reshape(c, a, b).transpose((1, 2, 0)).reshape(1, a, b, c)
        y = self.model(x)
        policise, value = y[0][0].numpy()[env.legal_action()], y[1][0][0]
        p_sum = np.sum(policise)
        policise /= p_sum if p_sum else 1
        return policise, value


class PVMCTS:
    def __init__(self, env, policy=0):
        self.env = env
        self.w = 0
        self.n = 0
        self.policy = policy
        self.child_nodes = []
    
    def max_child_node(self):
        C_PUCT = 1.0
        t = sum([child.n for child in self.child_nodes])
        pucb_lst = [(child.w /child.n if child.n else 0) + C_PUCT * child.policy * np.sqrt(t) / (1 + child.n) for child in self.child_nodes]
        return self.child_nodes[np.argmax(pucb_lst)]
    
    def evaluate(self, dualnet):
        if (value := self.env.is_done()) is not None:
            self.w += value
            self.n += 1
            return value
        
        if self.child_nodes:
            value = -self.max_child_node().evaluate(dualnet)

            self.w += value
            self.n += 1
            return value
        
        else:
            policies, value = dualnet.predict(self.env)
            self.w += value
            self.n += 1
            
            for action, policy in zip(self.env.legal_action(), policies):
                self.env.put(action)
                self.env.change_turn()
                self.child_nodes.append(PVMCTS(deepcopy(self.env), policy))
                self.env.change_turn()
                self.env.pb[action] = 0
            return value

    def boltzman(self, scores, t):
        if t:
            scores = scores ** (1 / t)
            return scores / np.sum(scores)
        else:
            action = np.argmax(scores)
            scores = np.zeros_like(scores)
            scores[action] = 1
            return scores
        
    def pvmcts_scores(self, dualnet, sim_num, t):
        for _ in range(sim_num):
            self.evaluate(dualnet)
        
        scores = np.array([child.n for child in self.child_nodes])
        self.__init__(self.env)
        return self.boltzman(scores, t)


class AlphaZero:
    def __init__(self, env, filter=128, residual_num=16, path=r"C:\Users\tomhi\python\tictactoe\model"):
        self.env = env
        self.path = path
        self.dualnet = DualNet((self.env.size, self.env.size, 2), self.env.action_size, filter, residual_num, path)
        self.pvmcts = PVMCTS(self.env)

    def take_action(self):
        scores = self.pvmcts.pvmcts_scores(self.dualnet, 10, 0)
        return np.random.choice(self.env.legal_action(), p=scores)
    
    def play(self, sim_num):
        history = []

        while True:
            scores = self.pvmcts.pvmcts_scores(self.dualnet, sim_num, 1)
            policies = np.zeros((self.env.action_size))
            legal_actions = self.env.legal_action()
            policies[legal_actions] = scores
            history.append([[self.env.pb, self.env.ob], policies, 0])
            action = np.random.choice(legal_actions, p=scores)
            self.env.put(action)
            self.env.change_turn()

            if (value := self.env.is_done()) is not None:
                break

        for data in history[::-1]:
            data[2] = value
            value *= -1
        self.env.reset()
        return history
    
    def self_play(self, game_num, sim_num, path):
        history = []
        for i in range(game_num):
            history.append(self.play(sim_num))
            #print(f"play {i+1}/{game_num}")

        with open(path, "wb") as f:
            pickle.dump(history, f) 

        k.clear_session()    

    def training(self, history, epoch_num):
        x_lst, y_policies, y_values = map(lambda x: np.array(x), zip(*history))
        a, b, c = self.dualnet.input_shape
        x_lst = x_lst.reshape(-1, c, a, b).transpose(0, 2, 3, 1)

        def scheduler(epoch):
            x = 0.001
            if epoch >= 50:
                x = 0.0005
            elif epoch >= 80:
                x = 0.000025
            return x
        lr_decay = LearningRateScheduler(scheduler)

        print_callback = LambdaCallback(
            on_epoch_begin=lambda epoch, logs: print(f"train {epoch}/{epoch_num}")
        )

        self.dualnet.model.compile(optimizer="adam", loss=["categorical_crossentropy", "mse"])
        self.dualnet.model.fit(
            x_lst, [y_policies, y_values], batch_size=64, epochs=epoch_num, verbose=0, 
            callbacks=[lr_decay, print_callback],
        )
        os.makedirs(self.path, exist_ok=True)
        self.dualnet.model.save(os.path.join(self.path, "best"))
        k.clear_session()        

