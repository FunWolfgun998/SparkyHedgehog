# --- START OF FILE wrappers.py ---

import gymnasium as gym
import numpy as np
from core.logger import sparky_logger  # INIETTATO IL LOGGER


class SparkyDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [[], ['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], ['B'], ['DOWN'], ['DOWN', 'B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for btn in action: arr[buttons.index(btn)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class SonicRAMWrapper(gym.Wrapper):
    def __init__(self, env, num_radar_objects=12):
        super().__init__(env)
        self.NUM_RADAR_OBJECTS = num_radar_objects
        total_shape = 22 + (self.NUM_RADAR_OBJECTS * 8)
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

        water_level = info.get('water_level', 4000)
        is_underwater = 1.0 if s_y > water_level else 0.0

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


class SparkyReward(gym.Wrapper):
    # --- CONFIGURAZIONE REWARD (Modifica qui e cambierà ovunque) ---
    REW_TIME = -0.15
    REW_WIN = 1000.0
    REW_RING_GAIN = 10.0
    REW_SPRING = 400.0
    REW_LOOP_SUCCESS = 300.0
    REW_BOSS_HIT = 300.0
    REW_ENEMY = 200.0
    REW_PROGRESS_MULT = 2.5

    PEN_ACTION_CHANGE = -0.05
    PEN_DAMAGE_BASE = 50.0
    PEN_RING_MULT = 15.0  # Moltiplicatore per ogni anello perso
    PEN_JUMP_LOOP = -20.0
    PEN_BACKTRACK = -2.0
    PEN_STUCK = -50.0
    PEN_DEATH_BASE = 200.0
    PEN_GLITCH_BASE = 300.0

    def __init__(self, env):
        super().__init__(env)
        self._init_vars()

    def _init_vars(self):
        self.max_x = self.prev_x = self.prev_g_speed = self.stuck_frames = 0
        self.prev_hp = 8
        self.prev_score = self.prev_rings = 0
        self.boss_initialized = self.loop_potential = self.reached_top = False
        self.first_step = True
        self.prev_action = 0
        self.prev_velocity_vett = 0

    def reset(self, **kwargs):
        self._init_vars()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        curr_x, curr_y = info.get('x', 0), info.get('y', 0)
        v_x, v_y = info.get('velocity_x', 0), info.get('velocity_y', 0)
        g_speed = abs(info.get('ground_speed', 0))
        angle = info.get('angle', 0)
        c_rings = info.get('rings', 0)
        curr_score = info.get('score', 0)
        curr_velocity_vett = (v_x ** 2 + v_y ** 2) ** 0.5
        status_byte = int(info.get('status', 0))
        in_air = (status_byte >> 1) & 1

        if self.first_step:
            self.max_x = self.prev_x = curr_x
            self.prev_g_speed = g_speed
            self.prev_action = action
            self.prev_velocity_vett = curr_velocity_vett
            self.prev_score = curr_score
            self.prev_hp = c_rings
            self.first_step = False

        # --- 1. IL COSTO DEL TEMPO ---
        step_reward = self.REW_TIME

        # --- 2. ACTION THRASHING ---
        if action != self.prev_action:
            step_reward += self.PEN_ACTION_CHANGE
        self.prev_action = action

        # --- 3. VITTORIA ---
        if info.get('level_end_bonus', 0) > 0:
            step_reward += self.REW_WIN
            self.stuck_frames = 0
            trunc = True
            sparky_logger.log("🏁 MISSION COMPLETE! Reward: +{r}", r=self.REW_WIN)

        # --- 4. DANNO E ANELLI DINAMICI ---
        if c_rings < self.prev_rings:
            rings_lost = self.prev_rings - c_rings
            # Calcolo dinamico: Base + (Moltiplicatore * Anelli)
            penalty = self.PEN_DAMAGE_BASE + (self.PEN_RING_MULT * rings_lost)
            step_reward -= penalty
            sparky_logger.log("💥 DAMAGE! Persi: {r} rings | Penalty: -{p}", r=rings_lost, p=penalty)
        elif c_rings > self.prev_rings:
            val = self.REW_RING_GAIN * (c_rings - self.prev_rings)
            step_reward += val
            sparky_logger.log("💰 RING! +{g})", g=val)
        self.prev_rings = c_rings

        # --- 5. MOLLE VETTORIALI ---
        is_spring_near = any(info.get(f'obj{i}_id', 0) == 65 for i in range(1, 9))
        delta_v = curr_velocity_vett - self.prev_velocity_vett
        if is_spring_near and v_y < -800:
            step_reward += self.REW_SPRING
            sparky_logger.log("🚀 TRAMPOLINO! Delta-V: {dv:.1f} | Reward: +{r}", dv=delta_v, r=self.REW_SPRING)
        self.prev_velocity_vett = curr_velocity_vett

        # --- 6. LOOP E IN-AIR LOGIC ---
        is_jump = action in [3, 4, 5]
        if not in_air:
            if 45 < angle < 215:
                self.loop_potential = True
                if 100 < angle < 150 and g_speed > 450:
                    self.reached_top = True
                if is_jump:
                    step_reward += self.PEN_JUMP_LOOP
            else:
                if self.loop_potential and self.reached_top:
                    if curr_x > self.max_x:
                        step_reward += self.REW_LOOP_SUCCESS
                        sparky_logger.log("🌀 LOOP SUCCESS! Reward: +{r}", r=self.REW_LOOP_SUCCESS)
                    self.loop_potential = self.reached_top = False

        # --- 7. PROGRESSIONE ORIZZONTALE ---
        if curr_x > self.max_x:
            val = (curr_x - self.max_x) * self.REW_PROGRESS_MULT
            step_reward += val
            self.max_x = curr_x
        elif curr_x < self.prev_x:
            step_reward += self.PEN_BACKTRACK

        # --- 8. BOSS EGGMAN E WATCHDOG ---
        is_boss = any(info.get(f'obj{i}_id', 0) == 61 for i in range(1, 9))
        if is_boss:
            self.stuck_frames = 0
            curr_hp = info.get('boss_hp', 8)
            if not self.boss_initialized:
                self.prev_hp, self.boss_initialized = curr_hp, True
            if curr_hp < self.prev_hp:
                step_reward += self.REW_BOSS_HIT
                sparky_logger.log("🎯 BOSS HIT! HP: {hp} | Reward: +{r}", hp=curr_hp, r=self.REW_BOSS_HIT)
            self.prev_hp = curr_hp
        else:
            self.boss_initialized = False
            if not trunc:
                if abs(curr_x - self.prev_x) < 2:
                    self.stuck_frames += 1
                else:
                    self.stuck_frames = 0
                if self.stuck_frames > 200:
                    step_reward += self.PEN_STUCK
                    trunc = True
                    sparky_logger.log("🛑 STUCK! Penalty: {p}", p=self.PEN_STUCK)

        # --- 9. NEMICI (SCORE) ---
        c_score = info.get('score', 0)
        if c_score > self.prev_score:
            if (c_score - self.prev_score) >= 100:
                step_reward += self.REW_ENEMY
                sparky_logger.log("👾 ENEMY DESTROYED! Reward: +{r}", r=self.REW_ENEMY)
        self.prev_score = c_score

        # --- 10. MORTE DINAMICA ---
        if term:
            rings_lost_at_death = self.prev_rings
            ring_penalty = self.PEN_RING_MULT * rings_lost_at_death

            is_hazard_near = any(info.get(f'obj{i}_id', 0) in [54, 64, 34, 75, 84] for i in range(1, 9))
            is_falling = curr_y > 900

            if not is_hazard_near and not is_falling:
                penalty = self.PEN_GLITCH_BASE + ring_penalty
                step_reward -= penalty
                sparky_logger.log("🚨 GLITCH DEATH! Rings: {r} | Penalty: -{p}", r=rings_lost_at_death, p=penalty)
            else:
                penalty = self.PEN_DEATH_BASE + ring_penalty
                step_reward -= penalty
                sparky_logger.log("💀 DEATH! Rings: {r} | Penalty: -{p}", r=rings_lost_at_death, p=penalty)

        self.prev_x, self.prev_g_speed = curr_x, g_speed

        return obs, step_reward / 100.0, term, trunc, info