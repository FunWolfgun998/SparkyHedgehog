import os
import random
import shutil
import gymnasium as gym
import stable_retro as retro
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import config
from core.wrappers import SparkyDiscretizer, SonicRAMWrapper, SparkyReward


def sync_retro_files():
    """Copia data.json, scenario.json e tutti i .state dentro la cartella di stable-retro"""
    retro_path = retro.__path__[0]
    game_path = os.path.join(retro_path, "data", "stable", config.GAME_NAME)
    os.makedirs(game_path, exist_ok=True)

    # 1. Sincronizza i file JSON (DNA del gioco)
    files_to_sync = ["data.json", "scenario.json"]
    for file in files_to_sync:
        src = os.path.join(config.ROOT_DIR, file)
        dst = os.path.join(game_path, file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"🔄 File JSON sincronizzato: {file}")

    # 2. Sincronizza i file .state (Checkpoint di addestramento)
    # Copiamo i file da train_states alla cartella di retro
    if os.path.exists(config.CUSTOM_STATES_DIR):
        state_files = [f for f in os.listdir(config.CUSTOM_STATES_DIR) if f.endswith('.state')]
        for f in state_files:
            src = os.path.join(config.CUSTOM_STATES_DIR, f)
            dst = os.path.join(game_path, f)
            shutil.copy(src, dst)
        if state_files:
            print(f"💾 {len(state_files)} stati sincronizzati nell'emulatore.")


class RandomResetWrapper(gym.Wrapper):
    def __init__(self, env, states):
        super().__init__(env)
        self.states = states
        self.state_index = random.randint(0, len(states) - 1)

    def reset(self, **kwargs):
        # 1. Sceglie il prossimo stato
        self.state_index = (self.state_index + 1) % len(self.states)
        current_state = str(self.states[self.state_index])

        # 2. Prepara l'emulatore a caricare quello stato
        try:
            self.env.unwrapped.load_state(current_state, retro.State.DEFAULT)
        except Exception as e:
            self.env.unwrapped.load_state(current_state)

        self._clear_ram_values()

        # Eseguiamo il reset dell'emulatore
        obs, info = self.env.reset(**kwargs)

        # TRUCCO: Eseguiamo 2 step con "nessuna azione" per stabilizzare la RAM
        # Questo pulisce i mirror interni del gioco che potrebbero ripristinare i valori
        for _ in range(2):
            obs, _, _, _, info = self.env.step(np.zeros(12, dtype=np.int8))
            self._clear_ram_values()  # Riaffermiamo il reset

        return obs, info
    def _clear_ram_values(self):
        """Forza i valori critici a zero nel motore del gioco"""
        self.env.unwrapped.data.set_value("rings", 0)
        self.env.unwrapped.data.set_value("score", 0)
        self.env.unwrapped.data.set_value("level_end_bonus", 0)

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
    # Sincronizza tutto prima di avviare i processi figli
    sync_retro_files()

    env_fns = [make_env(config.GAME_NAME, config.STATES, i) for i in range(config.NUM_ENVS)]
    venv = SubprocVecEnv(env_fns)
    return venv