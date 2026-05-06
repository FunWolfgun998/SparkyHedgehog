import stable_retro as retro
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import config
import shutil
import os
import random
import gymnasium as gym

# IMPORTA DALLA CARTELLA CORE
from core.wrappers import SparkyDiscretizer, SonicRAMWrapper, SparkyReward


def inject_data_json():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_path = os.path.join(base_path, "data.json")
    retro_path = os.path.join(os.path.dirname(retro.__file__), "data", "stable", config.GAME_NAME)
    dest_path = os.path.join(retro_path, "data.json")
    if os.path.exists(source_path):
        shutil.copyfile(source_path, dest_path)
    else:
        print(f"⚠️ ERRORE: data.json non trovato in {source_path}")


inject_data_json()


class RandomResetWrapper(gym.Wrapper):
    """Wrapper che cambia lo stato del gioco a ogni reset"""

    def __init__(self, env, states):
        super().__init__(env)
        self.states = states

    def reset(self, **kwargs):
        # Scegliamo uno stato a caso dalla lista
        new_state = random.choice(self.states)

        # Carichiamo lo stato. Usiamo 'DEFAULT' invece di 'GZ'
        # Se desse ancora errore, scrivi semplicemente: self.env.unwrapped.load_state(new_state)
        try:
            self.env.unwrapped.load_state(new_state, retro.State.DEFAULT)
        except Exception as e:
            # Fallback se la tua versione di retro preferisce solo il nome
            self.env.unwrapped.load_state(new_state)

        return self.env.reset(**kwargs)


def make_env(game, state_list, env_index=0):
    def _init():
        # Inizializziamo con uno stato qualsiasi (verrà sovrascritto dal wrapper al primo reset)
        env = retro.make(game=game, state=state_list[0], render_mode="rgb_array")

        # ORDINE DEI WRAPPER (Fondamentale)
        env = RandomResetWrapper(env, state_list)  # 1. Cambia lo stato
        env = SparkyDiscretizer(env)  # 2. Semplifica i tasti
        env = SparkyReward(env)  # 3. Gestisce i premi e il timeout
        env = SonicRAMWrapper(env)  # 4. Estrae i 40 sensori
        env = Monitor(env)  # 5. Registra i log
        return env

    return _init


def create_parallel_envs():
    # Passiamo la lista completa degli stati definita in config.py
    env_fns = [make_env(config.GAME_NAME, config.STATES, i) for i in range(config.NUM_ENVS)]
    return SubprocVecEnv(env_fns)