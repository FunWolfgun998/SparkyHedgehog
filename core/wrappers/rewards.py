import gymnasium as gym
from core.logger import sparky_logger


class SparkyReward(gym.Wrapper):
    REW_TIME = -0.15
    REW_WIN = 1000.0
    REW_RING_GAIN = 10.0
    REW_SPRING = 400.0
    REW_LOOP_SUCCESS = 300.0
    REW_BOSS_HIT = 300.0
    REW_ENEMY = 200.0
    REW_PROGRESS_MULT = 2.5
    REW_POWERUP = 200.0
    REW_BREATH = 300.0
    REW_CHECKPOINT = 500.0
    REW_1UP = 600.0
    REW_FIRST_RING = 60.0

    PEN_ACTION_CHANGE = -0.05
    PEN_DAMAGE_BASE = 50.0
    PEN_RING_MULT = 15.0
    PEN_JUMP_LOOP = -20.0
    PEN_BACKTRACK = -5.0
    PEN_STUCK = -50.0
    PEN_DEATH_BASE = 200.0
    PEN_GLITCH_BASE = 300.0
    PEN_VULNERABLE = -0.2
    PEN_WALL_PUSH = -0.5

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

        self.prev_shield = 0
        self.prev_invincible = 0
        self.prev_shoes = 0
        self.prev_air = 1800
        self.prev_lamppost = 0
        self.prev_lives = 3

        # NUOVO: Accumulatore per risolvere il bug dello score graduale
        self.score_accumulator = 0

    def reset(self, **kwargs):
        self._init_vars()
        return self.env.reset(**kwargs)

    def set_logger_state(self, state: bool):
        sparky_logger.console_enabled = state

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
            self.prev_shield = info.get('shield', 0)
            self.prev_invincible = info.get('invincible', 0)
            self.prev_shoes = info.get('shoes', 0)
            self.prev_air = info.get('air_timer', 1800)
            self.prev_lamppost = info.get('lamppost_id', 0)
            self.prev_lives = info.get('lives', 3)
            self.first_step = False

        step_reward = self.REW_TIME

        if action != self.prev_action: step_reward += self.PEN_ACTION_CHANGE
        self.prev_action = action

        if info.get('level_end_bonus', 0) > 0:
            step_reward += self.REW_WIN
            trunc = True
            sparky_logger.log("🏁 MISSION COMPLETE! Reward: +{r}", r=self.REW_WIN)

        # Gestione Anelli
        if c_rings == 0:
            step_reward += self.PEN_VULNERABLE
        elif c_rings > self.prev_rings:
            if self.prev_rings == 0:
                step_reward += self.REW_FIRST_RING
                anelli_extra = (c_rings - self.prev_rings) - 1
                if anelli_extra > 0: step_reward += (self.REW_RING_GAIN * anelli_extra)
            else:
                val = self.REW_RING_GAIN * (c_rings - self.prev_rings)
                step_reward += val
        self.prev_rings = c_rings

        # --- MODULO TRAMPOLINI CORRETTO ---
        # Ora che il radar legge da FFFFD800, i trampolini (ID 65) verranno trovati!
        is_spring_near = any(info.get(f'obj{i}_id', 0) == 65 for i in range(1, 25))
        delta_v = curr_velocity_vett - self.prev_velocity_vett
        if is_spring_near and v_y < -800:
            step_reward += self.REW_SPRING
            sparky_logger.log("🚀 TRAMPOLINO! Delta-V: {dv:.1f} | Reward: +{r}", dv=delta_v, r=self.REW_SPRING)
        self.prev_velocity_vett = curr_velocity_vett

        # --- MODULO LOOP CORRETTO (Fisica meno severa) ---
        is_jump = action in [3, 4, 5]
        if not in_air:
            if 45 < angle < 215:
                self.loop_potential = True
                if 100 < angle < 150 and g_speed > 350:  # Abbassato da 450 a 350 per essere realistico!
                    self.reached_top = True
                if is_jump:
                    step_reward += self.PEN_JUMP_LOOP
            else:
                if self.loop_potential and self.reached_top:
                    if curr_x > self.max_x:
                        step_reward += self.REW_LOOP_SUCCESS
                        sparky_logger.log("🌀 LOOP SUCCESS! Reward: +{r}", r=self.REW_LOOP_SUCCESS)
                    self.loop_potential = self.reached_top = False

        if curr_x > self.max_x:
            val = (curr_x - self.max_x) * self.REW_PROGRESS_MULT
            step_reward += val
            self.max_x = curr_x
        elif curr_x < self.prev_x:
            step_reward += self.PEN_BACKTRACK

        curr_lamppost = info.get('lamppost_id', 0)
        if curr_lamppost > self.prev_lamppost:
            step_reward += self.REW_CHECKPOINT
            self.prev_lamppost = curr_lamppost

        is_boss = any(info.get(f'obj{i}_id', 0) == 61 for i in range(1, 25))
        if is_boss:
            self.stuck_frames = 0
            curr_hp = info.get('boss_hp', 8)
            if not self.boss_initialized:
                self.prev_hp, self.boss_initialized = curr_hp, True
            if curr_hp < self.prev_hp:
                step_reward += self.REW_BOSS_HIT
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

        # --- MODULO NEMICI CORRETTO (Accumulatore Punti) ---
        score_diff = curr_score - self.prev_score
        if score_diff > 0:
            self.score_accumulator += score_diff
            if self.score_accumulator >= 100:
                step_reward += self.REW_ENEMY
                sparky_logger.log("👾 ENEMY DESTROYED! Reward: +{r}", r=self.REW_ENEMY)
                self.score_accumulator -= 100  # Sottrae 100 per contare nemici successivi
        self.prev_score = curr_score

        is_pushing = info.get('pushing_wall', 0)
        if is_pushing > 0 and in_air == 0:
            step_reward += self.PEN_WALL_PUSH

        curr_shield = info.get('shield', 0)
        curr_invinc = info.get('invincible', 0)
        curr_shoes = info.get('shoes', 0)
        if curr_shield > self.prev_shield: step_reward += self.REW_POWERUP
        if curr_invinc > self.prev_invincible: step_reward += self.REW_POWERUP
        if curr_shoes > self.prev_shoes: step_reward += self.REW_POWERUP
        self.prev_shield, self.prev_invincible, self.prev_shoes = curr_shield, curr_invinc, curr_shoes

        curr_air = info.get('air_timer', 1800)
        if (status_byte & 64):
            if curr_air < 720: step_reward -= 0.5
            if curr_air > self.prev_air + 600: step_reward += self.REW_BREATH
        self.prev_air = curr_air

        curr_lives = info.get('lives', 3)
        if curr_lives > self.prev_lives: step_reward += self.REW_1UP
        self.prev_lives = curr_lives

        if term:
            ring_penalty = self.PEN_RING_MULT * self.prev_rings
            is_hazard_near = any(info.get(f'obj{i}_id', 0) in [54, 64, 34, 75, 84] for i in range(1, 25))
            if not is_hazard_near and (curr_y <= info.get('y_limit_bottom', 4000)):
                step_reward -= (self.PEN_GLITCH_BASE + ring_penalty)
            else:
                step_reward -= (self.PEN_DEATH_BASE + ring_penalty)

        self.prev_x, self.prev_g_speed = curr_x, g_speed
        return obs, step_reward / 100.0, term, trunc, info