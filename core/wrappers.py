import gymnasium as gym
import numpy as np


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
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(40,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._extract_ram(info), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._extract_ram(info), reward, term, trunc, info

    def _extract_ram(self, info):
        s_x, s_y = info.get('x', 0), info.get('y', 0)
        ram = [s_x / 10000., s_y / 1000., info.get('velocity_x', 0) / 2000., info.get('velocity_y', 0) / 2000.,
               info.get('ground_speed', 0) / 2000., info.get('angle', 0) / 255., info.get('status', 0) / 255.]
        boss_active = any(info.get(f'obj{i}_id', 0) == 61 for i in range(1, 9))
        if boss_active:
            ram.extend([(info.get('boss_x', 0) - s_x) / 500., (info.get('boss_y', 0) - s_y) / 500.,
                        info.get('boss_hp', 8) / 8.])
        else:
            ram.extend([0., 0., 1.])
        objs = []
        SPRING, SPIKES, ENEMIES = 65, 54, [64, 34, 75, 84]
        for i in range(1, 9):
            o_id = info.get(f'obj{i}_id', 0)
            if o_id in [0, 1, 61]: continue
            dx, dy = info.get(f'obj{i}_x', 0) - s_x, info.get(f'obj{i}_y', 0) - s_y
            objs.append({'d': (dx ** 2 + dy ** 2) ** .5, 'dx': dx, 'dy': dy, 's': 1. if o_id == SPRING else 0.,
                         'i': 1. if o_id in [46, 37] else 0., 'dg': 1. if o_id == SPIKES else 0.,
                         'e': 1. if o_id in ENEMIES else 0.})
        objs = sorted(objs, key=lambda x: x['d'])[:5]
        while len(objs) < 5: objs.append({'dx': 0, 'dy': 0, 's': 0, 'i': 0, 'dg': 0, 'e': 0})
        for o in objs: ram.extend([o['dx'] / 500., o['dy'] / 500., o['s'], o['i'], o['dg'], o['e']])
        return np.array(ram, dtype=np.float32)


class SparkyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._init_vars()

    def _init_vars(self):
        self.max_x = self.prev_x = self.prev_g_speed = self.stuck_frames = 0
        self.prev_hp = 8
        self.prev_score = self.prev_rings = 0
        self.boss_initialized = self.loop_potential = self.reached_top = False
        self.first_step = True

    def reset(self, **kwargs):
        self._init_vars()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        curr_x, g_speed, angle = info.get('x', 0), abs(info.get('ground_speed', 0)), info.get('angle', 0)
        vel_y = info.get('velocity_y', 0)

        if self.first_step:
            self.max_x = self.prev_x = curr_x
            self.prev_g_speed = g_speed
            self.first_step = False

        # --- 1. IL COSTO DEL TEMPO ---
        step_reward = -0.15
        is_victory = info.get('level_end_bonus', 0) > 0
        is_boss = any(info.get(f'obj{i}_id', 0) == 61 for i in range(1, 9))

        # --- 2. VITTORIA ---
        if is_victory:
            step_reward += 1000.0
            self.stuck_frames = 0
            trunc = True  # Forza il reset per cambiare livello/stato
            print("🏁 MISSION COMPLETE!")

        # --- 3. DANNI E ANELLI ---
        c_rings = info.get('rings', 0)
        if c_rings > self.prev_rings:
            step_reward += 10.0
        elif c_rings < self.prev_rings:
            step_reward -= 40.0
            print("💥 DAMAGE TAKEN!")
        self.prev_rings = c_rings

        # --- 4. TRAMPOLINI (IL FIX DELL'ERRORE È QUI) ---
        # Cerchiamo l'ID 65 (Trampolino/Molla) nel dizionario info
        has_spring = any(info.get(f'obj{i}_id', 0) == 65 for i in range(1, 9))
        if has_spring and vel_y < -800:
            step_reward += 30.0
            print("🚀 TRAMPOLINO USATO (+30)!")

        # --- 5. LOOP (360°) E INERZIA ---
        is_jump = action in [3, 4, 5]
        if is_jump and g_speed < self.prev_g_speed and (self.prev_g_speed - g_speed) > 20:
            step_reward -= ((self.prev_g_speed - g_speed) * 0.1)

        if 15 < angle < 240:
            self.loop_potential = True
            if 100 < angle < 150 and g_speed > 450: self.reached_top = True
            if is_jump: step_reward -= 15.0
        else:
            if self.loop_potential and self.reached_top:
                if curr_x > self.max_x:
                    step_reward += 300.0
                    print("🌀 LOOP COMPLETATO (+300)")
                self.loop_potential = self.reached_top = False

            if curr_x > self.max_x:
                step_reward += (curr_x - self.max_x) * 2.5
                self.max_x = curr_x
            if curr_x < self.prev_x:
                step_reward -= 2.0

        # --- 6. BOSS EGGMAN ---
        if is_boss:
            self.stuck_frames = 0
            curr_hp = info.get('boss_hp', 8)
            if not self.boss_initialized: self.prev_hp, self.boss_initialized = curr_hp, True
            if curr_hp < self.prev_hp:
                step_reward += 300.0
                print(f"🎯 BOSS COLPITO! HP: {curr_hp}")
            self.prev_hp = curr_hp
        else:
            self.boss_initialized = False
            # Logica Anti-Stallo
            if not is_victory:
                if abs(curr_x - self.prev_x) < 2:
                    self.stuck_frames += 1
                else:
                    self.stuck_frames = 0
                if self.stuck_frames > 200:
                    step_reward -= 30.0
                    trunc = True

        # --- 7. NEMICI (SCORE) ---
        c_score = info.get('score', 0)
        if c_score > self.prev_score:
            if (c_score - self.prev_score) >= 100:
                step_reward += 150.0
                print("👾 NEMICO ELIMINATO (+150)")
        self.prev_score = c_score

        if term: step_reward -= 100.0
        self.prev_x, self.prev_g_speed = curr_x, g_speed

        return obs, step_reward / 100.0, term, trunc, info