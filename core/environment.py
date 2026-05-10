import os
import random
import gymnasium as gym
import stable_retro as retro
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import config
from core.wrappers import SparkyDiscretizer, SonicRAMWrapper, SparkyReward


class RandomResetWrapper(gym.Wrapper):
    def __init__(self, env, states):
        super().__init__(env)
        self.states = states
        # Contatore per forzare la rotazione degli stati
        self.state_index = random.randint(0, len(states) - 1)
        self.attempts_per_state = {state: 0 for state in states}

    def reset(self, **kwargs):
        # Scegliamo lo stato successivo in modo ciclico per garantire equità
        self.state_index = (self.state_index + 1) % len(self.states)
        current_state = self.states[self.state_index]

        self.attempts_per_state[current_state] += 1
        try:
            self.env.unwrapped.load_state(current_state, retro.State.DEFAULT)
        except:
            self.env.unwrapped.load_state(current_state)

            # --- 2. IL RESET FISICO DELLA RAM (Tabula Rasa) ---
            # Accediamo direttamente alla memoria del Sega Genesis
            # Questi nomi devono corrispondere a quelli nel tuo data.json
        self.env.unwrapped.data.set_value("rings", 0)
        self.env.unwrapped.data.set_value("score", 0)
        self.env.unwrapped.data.set_value("level_end_bonus", 0)

        # Opzionale: Resettiamo anche le vite a 3 per coerenza
        self.env.unwrapped.data.set_value("lives", 3)

        return self.env.reset(**kwargs)


def make_env(game, state_list, env_index=0):
    def _init():
        s = state_list[0] if state_list else config.STATE_NAME
        env = retro.make(game=game, state=s, render_mode="rgb_array")
        # Passiamo la lista stati al wrapper di reset
        env = RandomResetWrapper(env, state_list)
        env = SparkyDiscretizer(env)
        env = SonicRAMWrapper(env)
        env = SparkyReward(env)
        env = Monitor(env)
        return env

    return _init


def create_parallel_envs():
    env_fns = [make_env(config.GAME_NAME, config.STATES, i) for i in range(config.NUM_ENVS)]
    venv = SubprocVecEnv(env_fns)
    venv = VecFrameStack(venv, n_stack=4)
    return venv