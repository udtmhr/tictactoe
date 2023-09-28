import numpy as np
import copy

class Player:
    def __init__(self, env):
        self.env = env

    def take_action(self):
        while True:
            action = int(input())
            if self.env.check_action(action):
                return action
            print("そこにはおけません")
