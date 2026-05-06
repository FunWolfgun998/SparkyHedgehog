import gymnasium as gym
import numpy as np


class SparkyDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [
            [], ['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], ['B'], ['DOWN'], ['DOWN', 'B']
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


class SonicRAMWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 7 Fisica + 3 Boss + (5 Oggetti * 6 Dati radar) = 40
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(40,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._extract_ram(info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._extract_ram(info), reward, terminated, truncated, info

    def _extract_ram(self, info):
        s_x = info.get('x', 0)
        s_y = info.get('y', 0)

        ram = [
            s_x / 10000.0, s_y / 1000.0,
            info.get('velocity_x', 0) / 2000.0,
            info.get('velocity_y', 0) / 2000.0,
            info.get('ground_speed', 0) / 2000.0,
            info.get('angle', 0) / 255.0,
            info.get('status', 0) / 255.0
        ]

        boss_active = any(info.get(f'obj{i}_id', 0) == 61 for i in range(1, 9))
        if boss_active:
            ram.extend([
                (info.get('boss_x', 0) - s_x) / 500.0,
                (info.get('boss_y', 0) - s_y) / 500.0,
                info.get('boss_hp', 8) / 8.0
            ])
        else:
            ram.extend([0.0, 0.0, 1.0])

        objects = []
        SPRING, MONITOR, RING, SPIKES = 65, 46, 37, 54
        ENEMIES = [64, 34, 75, 84]

        for i in range(1, 9):
            o_id = info.get(f'obj{i}_id', 0)
            if o_id in [0, 1, 61]: continue
            dx = info.get(f'obj{i}_x', 0) - s_x
            dy = info.get(f'obj{i}_y', 0) - s_y
            dist = (dx ** 2 + dy ** 2) ** 0.5
            objects.append({
                'dist': dist, 'dx': dx, 'dy': dy,
                'is_spring': 1.0 if o_id == SPRING else 0.0,
                'is_item': 1.0 if o_id in [MONITOR, RING] else 0.0,
                'is_danger': 1.0 if o_id == SPIKES else 0.0,
                'is_enemy': 1.0 if o_id in ENEMIES else 0.0
            })

        objects = sorted(objects, key=lambda o: o['dist'])
        while len(objects) < 5:
            objects.append({'dist': 0, 'dx': 0, 'dy': 0, 'is_spring': 0, 'is_item': 0, 'is_danger': 0, 'is_enemy': 0})
        for obj in objects[:5]:
            ram.extend([obj['dx'] / 500.0, obj['dy'] / 500.0, obj['is_spring'], obj['is_item'], obj['is_danger'],
                        obj['is_enemy']])

        return np.array(ram, dtype=np.float32)


class SparkyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_x = 0
        self.prev_x = 0
        self.stuck_frames = 0
        self.prev_boss_hp = 8
        self.boss_initialized = False  # <--- AGGIUNTO
        self.first_step = True

    def reset(self, **kwargs):
        self.max_x, self.prev_x, self.stuck_frames, self.prev_boss_hp = 0, 0, 0, 8
        self.boss_initialized = False  # <--- AGGIUNTO
        self.first_step = True
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_x = info.get('x', 0)
        g_speed = info.get('ground_speed', 0)
        angle = info.get('angle', 0)

        if self.first_step:
            self.max_x, self.prev_x = curr_x, curr_x
            self.first_step = False

        delta_x = curr_x - self.prev_x
        self.prev_x = curr_x
        step_reward = 0

        # --- 1. LOGICA DEL PROGRESSO E INERZIA ---
        if 10 < angle < 245:
            step_reward += abs(g_speed) * 0.05
        else:
            if curr_x > self.max_x:
                step_reward += (curr_x - self.max_x) * 2.0
                self.max_x = curr_x

            if delta_x < 0:
                step_reward += max(delta_x, -5.0) * 0.5

                # --- 2. LOGICA COMBATTIMENTO BOSS (CORRETTA) ---
        # Controlliamo se Eggman è realmente presente
        is_boss_present = info.get('obj1_id', 0) == 61

        if is_boss_present:
            curr_hp = info.get('boss_hp', 8)
            if not self.boss_initialized:
                self.prev_boss_hp = curr_hp
                self.boss_initialized = True

            # Se la vita scende, diamo un premio enorme
            if curr_hp < self.prev_boss_hp and curr_hp >= 0:
                step_reward += 50.0
                print(f"🎯 BERSAGLIO COLPITO REALMENTE! Eggman HP: {curr_hp}")
            self.prev_boss_hp = curr_hp
        else:
            self.boss_initialized = False
            self.prev_boss_hp = 8

        # --- 3. ANTI-STALLO (TIMEOUT) ---
        if abs(delta_x) < 2 and abs(g_speed) < 100:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0

        if self.stuck_frames > 300:
            step_reward -= 20.0
            truncated = True
        if action == 2:  # Azione DESTRA
            if abs(info.get('velocity_x', 0)) < 10:
                step_reward -= 0.5  # Piccola punizione per stare contro il muro
        if terminated: step_reward -= 50.0

        return obs, step_reward / 100.0, terminated, truncated, info