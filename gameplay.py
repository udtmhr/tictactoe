from tictactoe import TicTacToe
from player.base_player import Player
from player.random_player import RandomPlayer, RandomPlayer2, RandomPlayer3
from player.qlearn import QLPlayer
from player.mcts import *

path = r'player\qtable\table3.4.npy'

env = TicTacToe()
com1 = QLPlayer(env, path)
com2 = Player(env)

while True:
    if env.is_done():
        break
    
    if env.is_first_player():
        action = com1.take_action()
    else:
        action = com2.take_action()
    env.put(action)
    env.change_turn()

    print(env)
    print(action)
    print()
