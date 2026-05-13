import gymnasium as gym
from core.logger import sparky_logger


class SparkyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # --- RICOMPENSE BASE (Fase 1) ---
        self.REW_TIME = -0.15  # Malus per forzarlo a muoversi
        self.REW_PROGRESS = 2.5  # Punti per ogni pixel spostato a destra
        self.PEN_BACKTRACK = -2.0  # Malus leggero se torna indietro
        self.REW_WIN = 1000.0  # Vittoria
        self.PEN_DEATH = 300.0  # Morte

        self.REW_RING = 20.0  # Premio per ogni anello preso
        self.PEN_DAMAGE = 100.0  # Malus fisso se viene colpito e perde anelli

        self._init_vars()

    def _init_vars(self):
        self.max_x = 0
        self.prev_x = 0
        self.prev_rings = 0
        self.first_step = True

    def reset(self, **kwargs):
        self._init_vars()
        return self.env.reset(**kwargs)

    def set_logger_state(self, state: bool):
        sparky_logger.console_enabled = state

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        curr_x = info.get('x', 0)
        c_rings = info.get('rings', 0)

        # Sincronizzazione iniziale
        if self.first_step:
            self.max_x = self.prev_x = curr_x
            self.prev_rings = c_rings
            self.first_step = False

        step_reward = self.REW_TIME

        # --- 1. PROGRESSIONE (Movimento) ---
        if curr_x > self.max_x:
            step_reward += (curr_x - self.max_x) * self.REW_PROGRESS
            self.max_x = curr_x
        elif curr_x < self.prev_x:
            step_reward += self.PEN_BACKTRACK

        self.prev_x = curr_x

        # --- 2. ECONOMIA ANELLI (Danni e Raccolta) ---
        if c_rings > self.prev_rings:
            anelli_presi = c_rings - self.prev_rings
            step_reward += self.REW_RING * anelli_presi
            sparky_logger.log("💰 RING PRESO! Totale: {c}", c=c_rings)

        elif c_rings < self.prev_rings and not term:
            # Se ha perso anelli ma non è morto, ha subito danno!
            step_reward -= self.PEN_DAMAGE
            sparky_logger.log("💥 DANNO SUBITO! Anelli azzerati.")

        self.prev_rings = c_rings

        # --- 3. VITTORIA E SCONFITTA ---
        if info.get('level_end_bonus', 0) > 0:
            step_reward += self.REW_WIN
            trunc = True
            sparky_logger.log("🏁 LIVELLO COMPLETATO!")

        if term:
            step_reward -= self.PEN_DEATH
            sparky_logger.log("💀 MORTO! (Spuntoni/Burrone/Nemico)")

        # Normalizzazione: Riduciamo la magnitudine per il PPO
        return obs, step_reward / 100.0, term, trunc, info