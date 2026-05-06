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
        self.max_x, self.prev_x, self.stuck_frames, self.prev_hp = 0, 0, 0, 8
        self.boss_initialized = self.loop_reached = False
        self.first_step = True

    def reset(self, **kwargs):
        self.max_x = self.prev_x = self.stuck_frames = 0
        self.prev_hp = 8
        self.boss_initialized = self.loop_reached = False
        self.first_step = True
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        curr_x, g_speed, angle = info.get('x', 0), info.get('ground_speed', 0), info.get('angle', 0)
        if self.first_step: self.max_x = self.prev_x = curr_x; self.first_step = False

        # 1. PENALITÀ TEMPO E IMMOBILITÀ
        step_reward = -0.1
        if abs(curr_x - self.prev_x) < 1 and abs(g_speed) < 50: step_reward -= 0.2

        # 2. LOGICA LOOP & SEMICERCHI
        is_jump = action in [3, 4, 5]
        if 20 < angle < 235:
            step_reward += abs(g_speed) * 0.1
            if 120 < angle < 140 and not self.loop_reached and abs(g_speed) > 400:
                step_reward += 100.0;
                self.loop_reached = True
                print("🌀 LOOP OK!")
            if is_jump: step_reward -= 5.0  # NON SALTARE SUI MURI!
        else:
            if curr_x > self.max_x: step_reward += (curr_x - self.max_x) * 2.0; self.max_x = curr_x
            if curr_x < self.prev_x: step_reward -= 1.5  # Backtracking punito

        # 3. BOSS
        if info.get('obj1_id', 0) == 61:
            curr_hp = info.get('boss_hp', 8)
            if not self.boss_initialized: self.prev_hp, self.boss_initialized = curr_hp, True
            if curr_hp < self.prev_hp: step_reward += 150.0
            self.prev_hp = curr_hp

        # 4. ANTI-STALLO
        if abs(curr_x - self.prev_x) < 2 and abs(g_speed) < 100:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0
        if self.stuck_frames > 200: step_reward -= 10.0; trunc = True

        if term: step_reward -= 100.0
        self.prev_x = curr_x
        return obs, step_reward / 100.0, term, trunc, info