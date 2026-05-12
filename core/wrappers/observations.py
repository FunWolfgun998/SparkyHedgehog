import gymnasium as gym
import numpy as np

class SonicRAMWrapper(gym.Wrapper):
    def __init__(self, env, num_radar_objects=12):
        super().__init__(env)
        self.NUM_RADAR_OBJECTS = num_radar_objects
        # FIX MATEMATICO: 20 Base + 4 Env + (12 * 9 Radar features) = 132
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

        # --- 1. FISICA CINEMATICA E STATUS (18 Dati) ---
        ram = [
            s_x / 10000.0,
            s_y / 4000.0,
            info.get('velocity_x', 0) / 2000.0,
            info.get('velocity_y', 0) / 2000.0,
            info.get('ground_speed', 0) / 2000.0,
            info.get('angle', 0) / 255.0,
            1.0 if info.get('rings', 0) > 0 else 0.0,
            info.get('air_timer', 1800) / 1800.0,
            (pit_limit - s_y) / 500.0,  # Distanza raw dal burrone
            1.0 if status & 1 else 0.0,  # Facing Left
            1.0 if status & 2 else 0.0,  # In Air
            1.0 if status & 4 else 0.0,  # Rolling
            1.0 if info.get('invincible', 0) > 0 else 0.0,
            1.0 if info.get('shield', 0) > 0 else 0.0,
            1.0 if info.get('shoes', 0) > 0 else 0.0,  # Speed Shoes
            1.0 if info.get('pushing_wall', 0) > 0 else 0.0, #se sta spingendo contro un muro
            info.get('boss_hp', 8) / 8.0,
            info.get('zone', 0) / 6.0,
            (info.get('boss_x', 0) - s_x) / 500.0 if any(info.get(f'obj{j}_id') == 61 for j in range(1, 17)) else 0.0,
            (info.get('boss_y', 0) - s_y) / 500.0 if any(info.get(f'obj{j}_id') == 61 for j in range(1, 17)) else 0.0
        ]

        # --- 2. SENSORI PERICOLO AMBIENTALE (4 Dati) ---
        dist_to_death = pit_limit - s_y
        pit_danger = max(0.0, 1.0 - (dist_to_death / 200.0))
        v_y = info.get('velocity_y', 0)
        falling_danger = 1.0 if (v_y > 200 and (status & 2)) else 0.0
        is_underwater = 1.0 if (status & 64) else 0.0

        cam_x = info.get('camera_x', 0)
        screen_pos_x = (s_x - cam_x) / 320.0

        ram.extend([pit_danger, falling_danger, is_underwater, screen_pos_x])
        # --- 3. RADAR OGGETTI ---
        CATEGORIES = {
            "LETHAL": [54, 62, 22, 31, 82, 111, 100, 106, 103],
            "PLATFORMS": [21, 24, 26, 28, 17, 60, 118, 122, 7],
            "ENEMIES": [34, 43, 64, 66, 75, 120, 83, 110, 35, 99, 80, 109, 113, 115],
            "PHYSICS": [65, 71, 119],
            "VITAL": [10],
            "ITEMS": [37, 46, 68, 59]
        }
        MAX_DIST_CULLING = ((190 ** 2) + (160 ** 2)) ** 0.5
        radar_objects = []
        for i in range(1, 17):
            o_id = info.get(f'obj{i}_id', 0)
            if o_id in [0, 1, 61]: continue

            dx = info.get(f'obj{i}_x', 0) - s_x
            dy = info.get(f'obj{i}_y', 0) - s_y
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist > MAX_DIST_CULLING: continue

            is_lethal, is_plat, is_enemy, is_phys, is_vital, is_item, base_score = 0, 0, 0, 0, 0, 0, 0
            if o_id in CATEGORIES["LETHAL"]: base_score, is_lethal = 100, 1
            elif o_id in CATEGORIES["PLATFORMS"]: base_score, is_plat = 95, 1
            elif o_id in CATEGORIES["VITAL"]: base_score, is_vital = 90, 1
            elif o_id in CATEGORIES["ENEMIES"]: base_score, is_enemy = 80, 1
            elif o_id in CATEGORIES["PHYSICS"]: base_score, is_phys = 70, 1
            elif o_id in CATEGORIES["ITEMS"]: base_score, is_item = 40, 1
            else:
                continue

            final_score = base_score * (1.0 - (dist / MAX_DIST_CULLING) * 0.6) * (1.0 if dx > -15 else 0.2)
            radar_objects.append({'id_norm': o_id / 255.0,'dx': dx / 500, 'dy': dy / 500, 'l': is_lethal, 'p': is_plat, 'e': is_enemy, 'is_phys': is_phys, 'v': is_vital, 'it': is_item,
                                  'score': final_score})

        radar_objects = sorted(radar_objects, key=lambda x: x['score'], reverse=True)[:self.NUM_RADAR_OBJECTS]

        while len(radar_objects) < self.NUM_RADAR_OBJECTS:
            radar_objects.append({'id_norm': 0,'dx': 0, 'dy': 0, 'l': 0, 'p': 0, 'e': 0, 'is_phys': 0, 'v': 0, 'it': 0})

        for o in radar_objects:
            ram.extend([o['id_norm'], o['dx'], o['dy'], o['l'], o['p'], o['e'], o['is_phys'], o['v'], o['it']])

        return np.array(ram, dtype=np.float32)
