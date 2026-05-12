import gymnasium as gym
import numpy as np

class SparkyDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [[], ['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], ['B'], ['DOWN'], ['DOWN', 'B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for btn in action: arr[buttons.index(btn)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()