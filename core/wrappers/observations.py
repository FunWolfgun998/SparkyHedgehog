import gymnasium as gym
import numpy as np


class SonicRAMWrapper(gym.Wrapper):
    def __init__(self, env, num_radar_objects=12):
        super().__init__(env)
        self.NUM_RADAR_OBJECTS = num_radar_objects
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

        # --- 1. FISICA CINEMATICA E STATUS (20 Dati) ---
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
            (info.get('boss_x', 0) - s_x) / 500.0 if any(info.get(f'obj{j}_id') == 61 for j in range(1, 17)) else 0.0,
            (info.get('boss_y', 0) - s_y) / 500.0 if any(info.get(f'obj{j}_id') == 61 for j in range(1, 17)) else 0.0
        ]

        # --- 2. SENSORI PERICOLO AMBIENTALE (4 Dati) ---
        dist_to_death = pit_limit - s_y
        pit_danger = max(0.0, 1.0 - (dist_to_death / 200.0))
        falling_danger = 1.0 if (info.get('velocity_y', 0) > 200 and (status & 2)) else 0.0

        cam_x = info.get('camera_x', 0)
        screen_pos_x = (s_x - cam_x) / 320.0
        ram.extend([pit_danger, falling_danger, 1.0 if (status & 64) else 0.0, screen_pos_x])

        info['ai_pit_danger'] = pit_danger
        info['ai_falling_danger'] = falling_danger
        info['ai_screen_x'] = screen_pos_x

        # --- 3. RADAR MODULARE (HUD vs AI) ---
        CATEGORIES = {
            "LETHAL": [54, 62, 22, 31, 82, 111, 100, 106, 103],
            "PLATFORMS": [21, 24, 26, 28, 17, 60, 118, 122, 7],
            "ENEMIES": [34, 43, 64, 66, 75, 120, 83, 110, 35, 99, 80, 109, 113, 115],
            "PHYSICS": [65, 71, 119],
            "VITAL": [10],
            "ITEMS": [37, 46, 68, 59]
        }

        MAX_DIST_AI = 350.0  # Raggio aumentato per l'IA
        radar_ai = []
        radar_hud = []

        for i in range(1, 17):
            o_id = info.get(f'obj{i}_id', 0)
            if o_id in [0, 1, 61]: continue  # Ignora Vuoti, Sonic e Boss

            dx = info.get(f'obj{i}_x', 0) - s_x
            dy = info.get(f'obj{i}_y', 0) - s_y
            dist = (dx ** 2 + dy ** 2) ** 0.5

            is_lethal = is_plat = is_enemy = is_phys = is_vital = is_item = is_unknown = 0
            base_score = 0

            # Ora gli oggetti sconosciuti NON vengono più ignorati
            if o_id in CATEGORIES["LETHAL"]:
                base_score, is_lethal = 100, 1
            elif o_id in CATEGORIES["PLATFORMS"]:
                base_score, is_plat = 95, 1
            elif o_id in CATEGORIES["VITAL"]:
                base_score, is_vital = 90, 1
            elif o_id in CATEGORIES["ENEMIES"]:
                base_score, is_enemy = 80, 1
            elif o_id in CATEGORIES["PHYSICS"]:
                base_score, is_phys = 70, 1
            elif o_id in CATEGORIES["ITEMS"]:
                base_score, is_item = 40, 1
            else:
                base_score, is_unknown = 10, 1  # Diamo priorità bassa ma lo tracciamo!

            final_score = base_score * (1.0 - min(dist / MAX_DIST_AI, 1.0) * 0.6) * (1.0 if dx > -15 else 0.2)

            obj_data = {
                'id': o_id, 'id_norm': o_id / 255.0,
                'raw_dx': dx, 'raw_dy': dy, 'dist': int(dist),
                'dx': dx / 500.0, 'dy': dy / 500.0,
                'l': is_lethal, 'p': is_plat, 'e': is_enemy,
                'is_phys': is_phys, 'v': is_vital, 'it': is_item, 'u': is_unknown,
                'score': final_score
            }

            # L'HUD riceve TUTTO quello che c'è nella RAM, a prescindere dalla distanza
            radar_hud.append(obj_data)

            # L'IA riceve solo ciò che è nel raggio d'azione e non sconosciuto
            if dist <= MAX_DIST_AI and not is_unknown:
                radar_ai.append(obj_data)

        # Ordina entrambi i radar per priorità (pericolosità)
        radar_hud = sorted(radar_hud, key=lambda x: x['score'], reverse=True)
        radar_ai = sorted(radar_ai, key=lambda x: x['score'], reverse=True)[:self.NUM_RADAR_OBJECTS]

        info['radar_debug'] = radar_hud  # Inietta i dati estesi per l'HUD

        # Aggiungi padding vuoto per l'IA se ci sono meno di 12 nemici
        while len(radar_ai) < self.NUM_RADAR_OBJECTS:
            radar_ai.append(
                {'id': 0, 'id_norm': 0, 'raw_dx': 0, 'raw_dy': 0, 'dist': 0, 'dx': 0, 'dy': 0, 'l': 0, 'p': 0, 'e': 0,
                 'is_phys': 0, 'v': 0, 'it': 0, 'u': 0})

        for o in radar_ai:
            ram.extend([o['id_norm'], o['dx'], o['dy'], o['l'], o['p'], o['e'], o['is_phys'], o['v'], o['it']])

        return np.array(ram, dtype=np.float32)