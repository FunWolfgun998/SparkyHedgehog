import gymnasium as gym
import numpy as np


class SonicRAMWrapper(gym.Wrapper):
    def __init__(self, env, num_radar_objects=12):
        super().__init__(env)
        self.NUM_RADAR_OBJECTS = num_radar_objects
        # 24 parametri base + (12 oggetti * 9 dati ognuno) = 132
        total_shape = 20 + (self.NUM_RADAR_OBJECTS * 9)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(total_shape,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._extract_ram(info), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._extract_ram(info), reward, term, trunc, info

    def _extract_ram(self, info):
        # Categorie ID per classificazione rapida (Boss Esclusi dal Radar)
        BOSS_IDS = [61, 90, 117, 118, 121, 126]
        CAT = {
            # 54: Spikes | 22: Harpoon | 23: Spiked Helix (GHZ) | 35: Buzz Missile | 72: Boss Ball
            # 87, 88: SYZ Spikes | 127: Boss Energy Ball
            "LETHAL": [54, 22, 23, 35, 72, 100, 106, 103, 111, 87, 88, 127],

            # 34: Buzz Bomber | 31: Crabmeat | 36: Newtron | 40: Motobug | 43: Chopper
            # 44: Jaws | 45: Burrobot | 64: Green Newtron | 32: Bomb | 80: Yadrin | 96,97: Orbinaut
            "ENEMIES": [34, 31, 36, 40, 43, 44, 45, 64, 32, 80, 96, 97],

            # Molle e Bumper
            "PHYSICS": [65, 41, 66, 30, 25, 71, 75, 76],

            # Checkpoint, Cartello fine livello, Bolla d'aria
            "VITAL": [10, 13, 14],

            # Anelli, Monitor TV
            "ITEMS": [37, 38],

            # Piattaforme, Ponti, Tronchi e la Capsula Prigione (62)
            "PLATFORMS": [6, 7, 8, 9, 11, 12, 15, 17, 18, 20, 21, 24, 26, 28, 33, 46, 47, 48, 49, 50, 52, 60, 62, 68,
                          77, 78, 79, 81, 82, 86, 89, 119, 122]
        }
        s_x = info.get('x', 0)
        s_y =info.get('y', 0)
        status = int(info.get('status', 0))
        camera_x = info.get('camera_x', 0)
        pit_limit = info.get('y_limit_bottom', 0)

        # --- 1. FISICA, STATUS E BOSS (20 Dati) ---
        boss_hp = 0
        boss_dx = 0
        boss_dy = 0
        for j in range(1, 61):
            o_id = info.get(f'obj{j}_id')
            if o_id in BOSS_IDS:
                    boss_hp = (info.get(f'obj{j}_hp', 0)) / 8.0
                    boss_dx = (info.get(f'obj{j}_x', 0) - s_x) / 500.0
                    boss_dy = (info.get(f'obj{j}_y', 0) - s_y) / 500.0
                    break

        ram = [
            s_x / 10000.0,           # 1
            s_y / 4000.0,            # 2
            info.get('velocity_x', 0) / 2000.0,      # 3: Vel Orizzontale
            info.get('velocity_y', 0) / 2000.0,      # 4: Vel Verticale
            info.get('ground_speed', 0) / 2000.0,    # 5: Vel Terreno (Cruciale per loop)
            info.get('angle', 0) / 255.0,            # 6: Angolo (0-255)
            1.0 if status & 1 else 0.0,              # 7: Facing Left
            1.0 if status & 2 else 0.0,              # 8: In Air
            1.0 if status & 4 else 0.0,              # 9: Rolling
            1.0 if info.get('rings', 0) > 0 else 0.0,  # 10: Binary "Has Armor"
            1.0 if info.get('invincible', 0) > 0 else 0.0, # 11
            1.0 if info.get('shield', 0) > 0 else 0.0,     # 12
            1.0 if info.get('shoes', 0) > 0 else 0.0,      # 13
            1.0 if info.get('pushing_wall', 0) > 0 else 0.0, # 14: Contro muro
            boss_hp,                                 # 15
            boss_dx,                                 # 16
            boss_dy,                                 # 17
            (s_x - camera_x) / 320.0,  # 18: Screen Relative X
            max(0.0, (pit_limit - s_y) / 500.0),  # 19: Pit Distance
            info.get('act', 0) / 2.0  # 20: Act Number (1, 2 o 3)
        ]

        # --- 3. RADAR OGGETTI (Esclusi Boss e Sonic) ---
        radar_ai = []
        MAX_DIST_CULLING = ((190 ** 2) + (180 ** 2)) ** 0.5  # Logica aggiornata

        for i in range(1, 61):
            o_id = info.get(f'obj{i}_id', 0)
            if o_id in [0, 1] or o_id in BOSS_IDS: continue

            dx = info.get(f'obj{i}_x', 0) - s_x
            dy = info.get(f'obj{i}_y', 0) - s_y
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist > MAX_DIST_CULLING: continue

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

            final_score = score * (1.0 - min(dist / MAX_DIST_CULLING, 1.0) * 0.6) * (1.0 if dx > -15 else 0.2)

            radar_ai.append({
                'id': o_id, 'dx': dx / 500, 'dy': dy / 500, 'dist': dist,
                'l': l, 'e': e, 'ph': ph, 'v': v, 'it': it, 'p': p,
                'score': final_score
            })

        radar_ai = sorted(radar_ai, key=lambda x: x['score'], reverse=True)[:self.NUM_RADAR_OBJECTS]
        info['ai_input_vector'] = ram  # Per il pannello sinistro HUD
        info['ai_radar_slots'] = radar_ai  # Per il pannello destro HUD

        # Riempimento finale del vettore (Padding tecnico per PPO)
        while len(radar_ai) < self.NUM_RADAR_OBJECTS:
            radar_ai.append({'id': 0, 'dx': 0, 'dy': 0, 'dist':999, 'l': 0, 'e': 0, 'ph': 0, 'v': 0, 'it': 0, 'p': 0})

        for o in radar_ai:
            ram.extend([o['id'] / 255.0, o['dx'], o['dy'], o['l'], o['e'], o['ph'], o['v'], o['it'], o['p']])

        return np.array(ram, dtype=np.float32)