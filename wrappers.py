# wrappers.py
import gymnasium as gym
import numpy as np
import cv2


class SparkyDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [
            [],  # 0: Fermo
            ['LEFT'],  # 1: Sinistra
            ['RIGHT'],  # 2: Destra
            ['LEFT', 'B'],  # 3: Sinistra + Salto
            ['RIGHT', 'B'],  # 4: Destra + Salto
            ['B'],  # 5: Salto sul posto
            ['DOWN'],  # 6: Giù
            ['DOWN', 'B'],  # 7: SPIN DASH
            ['START']  # 8: START
        ]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1, self.size, self.size), dtype=np.float32
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        frame = clahe.apply(frame)
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = np.power(frame, 1.2)
        frame = frame.reshape(self.size, self.size, 1)
        frame = np.transpose(frame, (2, 0, 1))
        return frame


class SparkyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_x = 0
        self.prev_x = 0
        self.stuck_frames = 0
        self.total_steps = 0
        self.first_step = True

    def reset(self, **kwargs):
        self.max_x = 0
        self.prev_x = 0
        self.stuck_frames = 0
        self.total_steps = 0
        self.first_step = True
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1

        curr_x = info.get('x', 0)
        curr_v = info.get('velocity_x', 0)

        if self.first_step:
            self.max_x = curr_x
            self.prev_x = curr_x
            self.first_step = False

        # Calcolo dello spostamento reale
        delta_x = curr_x - self.prev_x
        self.prev_x = curr_x

        step_reward = 0

        # --- 1. GESTIONE DEL MOVIMENTO E RECORD ---
        if curr_x > self.max_x:
            # Premio enorme se esplora nuovo territorio
            step_reward += (curr_x - self.max_x) * 2.0
            self.max_x = curr_x

        # Premio base se va avanti (costruisce rincorsa)
        if delta_x > 0:
            step_reward += delta_x * 0.5

        # PENALITÀ: Torna indietro o scivola giù da una rampa!
        if delta_x < 0:
            step_reward += delta_x * 1.0  # delta_x è negativo, quindi sottrae punti

        # --- 2. LA REGOLA "ANTI-SALTELLO" ---
        # Le azioni di salto sono: 3 (Sinistra+Salto), 4 (Destra+Salto), 5 (Salto fermo)
        if action in [3, 4, 5]:
            # A. Salto della disperazione (salta quando è quasi fermo)
            if abs(curr_v) < 3:
                step_reward -= 1.5  # Grossa sberla: "Non saltare se non hai rincorsa!"

            # B. Salto all'indietro (ha saltato ed è scivolato indietro)
            if delta_x < 0:
                step_reward -= 2.0  # Penalità gravissima!

        # --- 3. INCENTIVO ALLO SPIN DASH ---
        # L'azione 7 è DOWN + B (Spin Dash). Lo premiamo leggermente se lo carica da fermo
        if action == 7 and abs(curr_v) < 2:
            step_reward += 0.5  # "Bravo, stai caricando la mossa giusta!"

        # --- 4. GESTIONE STASI (Con grazia di 5 secondi) ---
        if self.total_steps > 300:
            # Bloccato: non va avanti e non ha velocità
            if delta_x <= 0 and curr_v <= 0:
                self.stuck_frames += 1
                step_reward -= 0.1
            else:
                self.stuck_frames = 0

            # Penalità muro: corre a destra ma non si muove
            if action in [2, 4] and delta_x <= 0:
                step_reward -= 0.2
        else:
            self.stuck_frames = 0

        # Reset forzato (3 secondi fermo)
        if self.stuck_frames > 180:
            step_reward -= 10.0
            truncated = True

        # --- 5. MORTE ---
        if terminated:
            step_reward -= 50.0

        return obs, step_reward, terminated, truncated, info

class RewardScaler(gym.RewardWrapper):
    def reward(self, reward):
        return reward / 100.0