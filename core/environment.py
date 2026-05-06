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
    """Copia data.json e scenario.json nella cartella di stable-retro"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    retro_path = os.path.join(os.path.dirname(retro.__file__), "data", "stable", config.GAME_NAME)

    for filename in ["data.json", "scenario.json"]:
        source = os.path.join(base_path, filename)
        dest = os.path.join(retro_path, filename)
        if os.path.exists(source):
            shutil.copyfile(source, dest)
            print(f"✅ Iniettato: {filename}")
        else:
            # Crea un file scenario vuoto se manca
            if filename == "scenario.json":
                with open(dest, 'w') as f: f.write('{"reward": {"variables": {}}}')


inject_data_json()


class RandomResetWrapper(gym.Wrapper):
    def __init__(self, env, states):
        super().__init__(env)
        self.states = states

    def reset(self, **kwargs):
        new_state = random.choice(self.states)
        # CORREZIONE: Usiamo retro.State.DEFAULT o passiamo solo il nome
        try:
            self.env.unwrapped.load_state(new_state, retro.State.DEFAULT)
        except:
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