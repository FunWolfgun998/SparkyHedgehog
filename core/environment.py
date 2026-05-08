import sys
import os
import shutil
import random
import gymnasium as gym
import stable_retro as retro
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import config

from core.wrappers import SparkyDiscretizer, SonicRAMWrapper, SparkyReward


def sync_retro_files():
    """Inietta i file JSON e gli stati. Chiamata SOLO una volta dal main."""
    retro_path = os.path.join(os.path.dirname(retro.__file__), "data", "stable", config.GAME_NAME)

    # Iniezione JSON
    for filename in ["data.json", "scenario.json"]:
        source = os.path.join(config.ROOT_DIR, filename)
        dest = os.path.join(retro_path, filename)
        if os.path.exists(source):
            shutil.copyfile(source, dest)
            print(f"✅ Sincronizzato: {filename}")

    # Sincronizzazione Stati
    state_files = config.get_state_files()
    for f in state_files:
        shutil.copyfile(f, os.path.join(retro_path, os.path.basename(f)))
    print(f"✅ {len(state_files)} stati pronti nell'emulatore.")


class RandomResetWrapper(gym.Wrapper):
    def __init__(self, env, states):
        super().__init__(env)
        self.states = states

    def reset(self, **kwargs):
        if self.states:
            new_state = random.choice(self.states)
            try:
                self.env.unwrapped.load_state(new_state, retro.State.DEFAULT)
            except:
                self.env.unwrapped.load_state(new_state)
        return self.env.reset(**kwargs)


def make_env(game, state_list, env_index=0):
    def _init():
        # Usa lo stato di default se non ci sono stati custom
        s = state_list[0] if state_list else config.STATE_NAME
        env = retro.make(game=game, state=s, render_mode="rgb_array")
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