import gymnasium as gym
import numpy as np


class SonicRAMWrapper(gym.Wrapper):
    def __init__(self, env, num_radar_objects=12):
        super().__init__(env)
        self.NUM_RADAR_OBJECTS = num_radar_objects
        # 24 parametri base + (12 oggetti * 9 dati ognuno) = 132
        total_shape = 24 + (self.NUM_RADAR_OBJECTS * 9)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(total_shape,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._extract_ram(info), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._extract_ram(info), reward, term, trunc, info

    def _extract_ram(self, info):
        s_x, s_y = info.get('x', 0), info.get('y', 0)
        status = int(info.get('status', 0))
        pit_limit = info.get('y_limit_bottom', 0)

        # Categorie ID per classificazione rapida (Boss Esclusi dal Radar)
        BOSS_IDS = [61, 90, 117, 118, 121, 126]
        CAT = {
            "LETHAL": [54, 22, 62, 100, 106, 103, 111, 21, 28, 31, 87, 88, 127],
            "ENEMIES": [34, 35, 36, 40, 43, 44, 45, 64, 80, 96, 97],
            "PHYSICS": [65, 41, 66, 30, 25, 71, 72, 75, 76],
            "VITAL": [10, 14, 13],
            "ITEMS": [37, 38],
            "PLATFORMS": [6, 7, 8, 9, 11, 12, 15, 17, 18, 20, 23, 24, 26, 33, 46, 47, 48, 49, 50, 52, 60, 68, 77, 78,
                          79, 81, 82, 86, 89, 119, 122]
        }

        # --- 1. FISICA, STATUS E BOSS (20 Dati) ---
        boss_present = any(info.get(f'obj{j}_id') in BOSS_IDS for j in range(1, 61))

        ram = [
            s_x / 10000.0, s_y / 4000.0,
            info.get('velocity_x', 0) / 2000.0, info.get('velocity_y', 0) / 2000.0,
            info.get('ground_speed', 0) / 2000.0, info.get('angle', 0) / 255.0,
            1.0 if info.get('rings', 0) > 0 else 0.0, info.get('air_timer', 1800) / 1800.0,
            (pit_limit - s_y) / 500.0,
            1.0 if status & 1 else 0.0, 1.0 if status & 2 else 0.0, 1.0 if status & 4 else 0.0,
            1.0 if info.get('invincible', 0) > 0 else 0.0, 1.0 if info.get('shield', 0) > 0 else 0.0,
            1.0 if info.get('shoes', 0) > 0 else 0.0, 1.0 if info.get('pushing_wall', 0) > 0 else 0.0,
            info.get('boss_hp', 8) / 8.0, info.get('zone', 0) / 6.0,
            (info.get('boss_x', 0) - s_x) / 500.0 if boss_present else 0.0,
            (info.get('boss_y', 0) - s_y) / 500.0 if boss_present else 0.0
        ]

        # --- 2. AMBIENTE (4 Dati) ---
        dist_to_death = pit_limit - s_y
        pit_danger = max(0.0, 1.0 - (dist_to_death / 200.0))
        falling_danger = 1.0 if (info.get('velocity_y', 0) > 200 and (status & 2)) else 0.0
        cam_x = info.get('camera_x', 0)
        ram.extend([pit_danger, falling_danger, 1.0 if (status & 64) else 0.0, (s_x - cam_x) / 320.0])

        # --- 3. RADAR OGGETTI (Esclusi Boss e Sonic) ---
        radar_ai = []
        max_dist = 350.0

        for i in range(1, 61):
            o_id = info.get(f'obj{i}_id', 0)
            if o_id in [0, 1] or o_id in BOSS_IDS: continue

            dx = info.get(f'obj{i}_x', 0) - s_x
            dy = info.get(f'obj{i}_y', 0) - s_y
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist > max_dist: continue

            # Flags di categoria
            l, e, ph, v, it, p = 0, 0, 0, 0, 0, 0
            score = 0

            if o_id in CAT["LETHAL"]:
                score, l = 100, 1
            elif o_id in CAT["ENEMIES"]:
                score, e = 85, 1
            elif o_id in CAT["PHYSICS"]:
                score, ph = 80, 1
            elif o_id in CAT["VITAL"]:
                score, v = 75, 1
            elif o_id in CAT["ITEMS"]:
                score, it = 60, 1
            elif o_id in CAT["PLATFORMS"]:
                score, p = 40, 1
            else:
                continue

            radar_ai.append({
                'id': o_id, 'dx': dx / 500, 'dy': dy / 500, 'dist': dist,
                'l': l, 'e': e, 'ph': ph, 'v': v, 'it': it, 'p': p,
                'score': score * (1.0 - (dist / max_dist) * 0.6)
            })

        radar_ai = sorted(radar_ai, key=lambda x: x['score'], reverse=True)[:self.NUM_RADAR_OBJECTS]
        info['ai_input_vector'] = ram  # Per il pannello sinistro HUD
        info['ai_radar_slots'] = radar_ai  # Per il pannello destro HUD

        # Riempimento finale del vettore (Padding tecnico per PPO)
        while len(radar_ai) < self.NUM_RADAR_OBJECTS:
            radar_ai.append({'id': 0, 'dx': 0, 'dy': 0, 'l': 0, 'e': 0, 'ph': 0, 'v': 0, 'it': 0, 'p': 0})

        for o in radar_ai:
            ram.extend([o['id'] / 255.0, o['dx'], o['dy'], o['l'], o['e'], o['ph'], o['v'], o['it'], o['p']])

        return np.array(ram, dtype=np.float32)