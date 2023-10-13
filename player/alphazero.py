import os
import numpy as np
from copy import deepcopy
from tensorflow.keras.layers import Dense, Conv2D, Input, Activation, Add, BatchNormalization, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as k
from mcs import MCTSPlayer


class DualNet:
    def __init__(self, input_shape, output_shape, filter, residual_num):
        self.filter = filter
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.residual_num = residual_num
        self.make_dual_net()
        self.model = load_model(r"./model/best")

    def conv(self):
        return Conv2D(self.filter, 3, padding="same", use_bias=False, 
                    kernel_initializer="he_normal", kernel_regularizer=l2(0.0005))
        
    def residual_block(self):
        def f(x):
            sc = x
            x = self.conv()(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = self.conv()(x)
            x = BatchNormalization()(x)
            x = Add()([x, sc])
            x = Activation("relu")(x)
            return x
        return f

    def make_dual_net(self):
        if os.path.exists(r"./model/best"):
            return
        
        input = Input(shape=self.input_shape)

        x = self.conv()(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        for _ in range(self.residual_num):
            x = self.residual_block()(x)

        policy = Conv2D(2, 1, padding="same", kernel_initializer="he_normal", 
                        kernel_regularizer=l2(0.0005))(x)
        policy = BatchNormalization()(policy)
        policy = Activation("relu")(policy)
        policy = Flatten()(policy)
        policy = Dense(self.output_shape, activation="softmax", kernel_regularizer=l2(0.0005), name="policy")(policy)
        
        value = Conv2D(1, 1, padding="same", kernel_initializer="he_normal", 
                        kernel_regularizer=l2(0.0005))(x)
        value = BatchNormalization()(value)
        value = Activation("relu")(value)
        value = Flatten()(value)
        value = Dense(self.filter, activation="relu", kernel_regularizer=l2(0.0005))(value)
        value = Dense(1, activation="tanh", kernel_regularizer=l2(0.0005), name="value")(value)

        model = Model(inputs=input, outputs=[policy, value])

        model.compile(loss=["categorical_crossentropy", "mse"], optimizer="adam")
        os.makedirs(r"./model/", exist_ok=True)
        model.save(r"./model/best")

        k.clear_session()
        del model

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
    def __init__(self, env, dualnet, policy=0):
        self.env = env
        self.w = 0
        self.n = 0
        self.dualnet = dualnet
        self.policy = policy
        self.child_nodes = []
    
    def max_child_node(self):
        C_PUCT = 1.0
        t = sum([child.n for child in self.child_nodes])
        pucb_lst = [(child.w /child.n if child.n else 0) + C_PUCT * child.policy * np.sqrt(t) / (1 + child.n) for child in self.child_nodes]
        return self.child_nodes[np.argmax(pucb_lst)]
    
    def evaluate(self):
        if (value := self.env.is_done()) is not None:
            self.w += value
            self.n += 1
            return value
        
        if self.child_nodes:
            value = -self.max_child_node().evaluate()

            self.w += value
            self.n += 1
            return value
        
        else:
            policies, value = self.dualnet.predict(self.env)
            self.w += value
            self.n += 1
            
            for action, policy in zip(self.env.legal_action(), policies):
                self.env.put(action)
                self.env.change_turn()
                self.child_nodes.append(PVMCTS(deepcopy(self.env), self.dualnet, policy))
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
        
    def pvmcts_scores(self, sim_num, t):
        for _ in range(sim_num):
            self.evaluate()
        
        scores = np.array([child.n for child in self.child_nodes])
        return self.boltzman(scores, t)      
    
    def pvmcts_action(self, sim_num, t):
        scores = self.pvmcts_scores(sim_num, t)
        self.__init__(self.env, self.dualnet)
        return np.random.choice(self.env.legal_action(), p=scores)





if __name__ == "__main__":
    import sys
    sys.path.append(r'C:\Users\tomhi\python\tictactoe')
    from tictactoe import TicTacToe

    dualnet = DualNet((3, 3, 2), 9, 128, 16)
    env = TicTacToe()
    pvmcts = PVMCTS(env, dualnet)
    while True:
        if env.is_done() is not None:
            break
        action = pvmcts.pvmcts_action(10, 1.0)
        env.put(action)
        env.change_turn()
        print(env)
