import stable_retro as retro
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import config
import shutil
import os
import random
import gymnasium as gym

from core.wrappers import SparkyDiscretizer, SonicRAMWrapper, SparkyReward


def sync_retro_files():
    """Inietta data.json, scenario.json e copia TUTTI i tuoi stati custom nell'emulatore"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    retro_path = os.path.join(os.path.dirname(retro.__file__), "data", "stable", config.GAME_NAME)

    # 1. Copia i file di configurazione JSON
    for filename in ["data.json", "scenario.json"]:
        source = os.path.join(base_path, filename)
        dest = os.path.join(retro_path, filename)
        if os.path.exists(source):
            shutil.copyfile(source, dest)
        elif filename == "scenario.json":
            with open(dest, 'w') as f:
                f.write('{"reward": {"variables": {}}}')

    # 2. Sincronizza gli Stati dalla tua cartella custom all'emulatore
    for state_file in config.STATE_FILES:
        dest_state = os.path.join(retro_path, os.path.basename(state_file))
        shutil.copyfile(state_file, dest_state)


sync_retro_files()


class RandomResetWrapper(gym.Wrapper):
    def __init__(self, env, states):
        super().__init__(env)
        self.states = states

    def reset(self, **kwargs):
        new_state = random.choice(self.states)
        try:
            self.env.unwrapped.load_state(new_state, retro.State.DEFAULT)
        except:
            self.env.unwrapped.load_state(new_state)
        return self.env.reset(**kwargs)


def make_env(game, state_list, env_index=0):
    def _init():
        env = retro.make(game=game, state=state_list[0], render_mode="rgb_array")
        env = RandomResetWrapper(env, state_list)
        env = SparkyDiscretizer(env)
        env = SparkyReward(env)
        env = SonicRAMWrapper(env)
        env = Monitor(env)
        return env

    return _init


def create_parallel_envs():
    env_fns = [make_env(config.GAME_NAME, config.STATES, i) for i in range(config.NUM_ENVS)]
    return SubprocVecEnv(env_fns)