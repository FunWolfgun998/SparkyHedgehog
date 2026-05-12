import os
import random
import shutil
import gymnasium as gym
import stable_retro as retro
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import config
from core.wrappers import SparkyDiscretizer, SonicRAMWrapper, SparkyReward


# --- FUNZIONE DI SINCRONIZZAZIONE JSON ---
def sync_retro_files():
    """Copia data.json e scenario.json dentro la cartella di stable-retro"""
    retro_path = retro.__path__[0]
    game_path = os.path.join(retro_path, "data", "stable", config.GAME_NAME)

    files_to_sync = ["data.json", "scenario.json"]
    for file in files_to_sync:
        src = os.path.join(config.ROOT_DIR, file)
        dst = os.path.join(game_path, file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"🔄 File sincronizzato nell'emulatore: {file}")


class RandomResetWrapper(gym.Wrapper):
    def __init__(self, env, states):
        super().__init__(env)
        self.states = states
        self.state_index = random.randint(0, len(states) - 1)
        self.attempts_per_state = {state: 0 for state in states}

    def reset(self, **kwargs):
        self.state_index = (self.state_index + 1) % len(self.states)
        current_state = self.states[self.state_index]
        self.attempts_per_state[current_state] += 1

        try:
            self.env.unwrapped.load_state(current_state, retro.State.DEFAULT)
        except:
            self.env.unwrapped.load_state(current_state)

        # Tabula Rasa Sicura
        self.env.unwrapped.data.set_value("rings", 0)
        self.env.unwrapped.data.set_value("score", 0)
        self.env.unwrapped.data.set_value("level_end_bonus", 0)
        self.env.unwrapped.data.set_value("lives", 3)

        return self.env.reset(**kwargs)


def make_env(game, state_list, env_index=0):
    def _init():
        s = state_list[0] if state_list else config.STATE_NAME
        env = retro.make(game=game, state=s, render_mode="rgb_array")
        env = RandomResetWrapper(env, state_list)
        env = SparkyDiscretizer(env)
        env = SonicRAMWrapper(env)
        env = SparkyReward(env)
        env = Monitor(env)
        return env

    return _init


def create_parallel_envs():
    # Prima di avviare i core paralleli, iniettiamo il DNA (data.json)
    sync_retro_files()

    env_fns = [make_env(config.GAME_NAME, config.STATES, i) for i in range(config.NUM_ENVS)]
    venv = SubprocVecEnv(env_fns)
    # RIMOSSO VecFrameStack: Non serve più alla nostra MlpPolicy da 188 variabili!
    return venv