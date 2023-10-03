import numpy as np


class TicTacToe:
    def __init__(self, size=3):
        self.size = size
        self.turn = 1
        self.pb = np.zeros(size * size, dtype=np.int64)
        self.ob = np.zeros(size * size, dtype=np.int64)
        self.action_size = self.size ** 2
        self.state_size = 3 ** self.action_size
    
    def reset(self):
        self.__init__(self.size)

    def check_action(self, action):
        if 0 <= action < self.size ** 2:
            return self.pb[action] == 0 and self.ob[action] == 0
        return False
    
    def put(self, action):
        self.pb[action] += 1
    
    def is_win(self):
        board = self.pb.reshape((self.size, self.size))
        return (np.all(board == 1, axis=0).any() or 
                np.all(board == 1, axis=1).any() or
                np.all(board.diagonal() == 1) or
                np.all(board[:, ::-1].diagonal() == 1))
    
    def is_lose(self):
        board = self.ob.reshape((self.size, self.size))
        return (np.all(board == 1, axis=0).any() or 
                np.all(board == 1, axis=1).any() or
                np.all(board.diagonal() == 1) or
                np.all(board[:, ::-1].diagonal() == 1))

    def is_drow(self):
        return np.sum(self.pb + self.ob) == 9
    
    def is_done(self):
        return self.is_win() or self.is_lose() or self.is_drow()
    
    def legal_action(self):
        return np.where(self.pb + self.ob == 0)[0]
    
    def change_turn(self):
        self.turn *= -1
        self.pb, self.ob = self.ob, self.pb

    def is_first_player(self):
        return self.turn == 1
    
    def reward(self):
        if self.is_lose():
            return -1
        if self.is_win():
            return 1
        else:
            return 0

    def get_state(self):
        o_board, x_board = (self.pb, self.ob) if self.is_first_player() else (self.ob, self.pb)
        o_board = np.where(o_board == 1)[0]
        x_board = np.where(x_board == 1)[0]
        func = lambda x: 3 ** x
        return func(o_board).sum() + func(x_board).sum() * 2
    
    def __str__(self):
        str = ""
        ox = ["_", "O", "X"]
        for i in range(self.size ** 2):
            if self.pb[i]:
                str += ox[self.turn]
            elif self.ob[i]:
                str += ox[self.turn * -1]
            else:
                str += ox[0]
            if i % self.size == 2:
                str += "\n"
        return str
    
if __name__ == "__main__":
    game = TicTacToe()
    for i in [1, 2, 7]:
        game.put(i)
    game.change_turn()
    for i in [0, 5, 6, 8]:
        game.put(i)
    print(game)
    print(game.get_state())
