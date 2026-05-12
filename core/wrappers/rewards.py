import gymnasium as gym
from core.logger import sparky_logger


class SparkyReward(gym.Wrapper):
    # --- CONFIGURAZIONE REWARD ORIGINALE ---
    REW_TIME = -0.15
    REW_WIN = 1000.0
    REW_RING_GAIN = 10.0
    REW_SPRING = 400.0
    REW_LOOP_SUCCESS = 300.0
    REW_BOSS_HIT = 300.0
    REW_ENEMY = 200.0
    REW_PROGRESS_MULT = 2.5
    REW_POWERUP = 200.0  # Scudi, Scarpe, Invincibilità
    REW_BREATH = 300.0  # Bolla d'aria sott'acqua
    REW_CHECKPOINT = 500.0  # Lamppost superato
    REW_1UP = 600.0  # Vita extra ottenuta
    REW_FIRST_RING = 60.0

    PEN_ACTION_CHANGE = -0.05
    PEN_DAMAGE_BASE = 50.0
    PEN_RING_MULT = 15.0
    PEN_JUMP_LOOP = -20.0
    PEN_BACKTRACK = -2.0
    PEN_STUCK = -50.0
    PEN_DEATH_BASE = 200.0
    PEN_GLITCH_BASE = 300.0
    PEN_VULNERABLE = -0.2
    PEN_WALL_PUSH = -0.5  # Malus continuo se spinge contro un muro


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

        # Inizializzazione nuove variabili di tracciamento
        self.prev_shield = 0
        self.prev_invincible = 0
        self.prev_shoes = 0
        self.prev_air = 1800
        self.prev_lamppost = 0
        self.prev_lives = 3

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

            # Sincronizzazione al primo frame per evitare falsi positivi
            self.prev_shield = info.get('shield', 0)
            self.prev_invincible = info.get('invincible', 0)
            self.prev_shoes = info.get('shoes', 0)
            self.prev_air = info.get('air_timer', 1800)
            self.prev_lamppost = info.get('lamppost_id', 0)
            self.prev_lives = info.get('lives', 3)

            self.first_step = False

        step_reward = self.REW_TIME

        # --- MODULO 1: Penalità cambio azione e Vittoria ---
        if action != self.prev_action:
            step_reward += self.PEN_ACTION_CHANGE
        self.prev_action = action

        if info.get('level_end_bonus', 0) > 0:
            step_reward += self.REW_WIN
            self.stuck_frames = 0
            trunc = True
            sparky_logger.log("🏁 MISSION COMPLETE! Reward: +{r}", r=self.REW_WIN)

        # --- MODULO 2: Gestione Anelli e Danni ---
        if c_rings == 0:
            step_reward += self.PEN_VULNERABLE
        elif c_rings < self.prev_rings:
            if self.prev_rings == 0:
                # Diamo il bonus speciale per il primo
                step_reward += self.REW_FIRST_RING
                anelli_extra = (c_rings - self.prev_rings) - 1
                if anelli_extra > 0:
                    step_reward += (self.REW_RING_GAIN * anelli_extra)

                sparky_logger.log("🛡️ PRIMO ANELLO (Salvavita)! Bonus: +{b} | Totale: {c}", b=self.REW_FIRST_RING, c=c_rings)
            else:
                # Guadagno standard per anelli successivi
                val = self.REW_RING_GAIN * (c_rings - self.prev_rings)
                step_reward += val
                sparky_logger.log("💰 RING! +{g}", g=val)
        self.prev_rings = c_rings

        # --- MODULO 3: Trampolini e Inerzia ---
        is_spring_near = any(info.get(f'obj{i}_id', 0) == 65 for i in range(1, 17))
        delta_v = curr_velocity_vett - self.prev_velocity_vett
        if is_spring_near and v_y < -800:
            step_reward += self.REW_SPRING
            sparky_logger.log("🚀 TRAMPOLINO! Delta-V: {dv:.1f} | Reward: +{r}", dv=delta_v, r=self.REW_SPRING)
        self.prev_velocity_vett = curr_velocity_vett

        # --- MODULO 4: Dinamica dei Loop ---
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

        # --- MODULO 5: Progressione, Backtrack e Checkpoint ---
        if curr_x > self.max_x:
            val = (curr_x - self.max_x) * self.REW_PROGRESS_MULT
            step_reward += val
            self.max_x = curr_x
        elif curr_x < self.prev_x:
            step_reward += self.PEN_BACKTRACK

        curr_lamppost = info.get('lamppost_id', 0)
        if curr_lamppost > self.prev_lamppost:
            step_reward += self.REW_CHECKPOINT
            sparky_logger.log("🚩 CHECKPOINT SUPERATO! +{r}", r=self.REW_CHECKPOINT)
            self.prev_lamppost = curr_lamppost

        # --- MODULO 6: Combattimento e Nemici ---
        is_boss = any(info.get(f'obj{i}_id', 0) == 61 for i in range(1, 17))
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

        c_score = info.get('score', 0)
        if c_score > self.prev_score:
            if (c_score - self.prev_score) >= 100:
                step_reward += self.REW_ENEMY
                sparky_logger.log("👾 ENEMY DESTROYED! Reward: +{r}", r=self.REW_ENEMY)
        self.prev_score = c_score

        # --- MODULO 7: Anti-Muro (Pushing Wall) ---
        is_pushing = info.get('pushing_wall', 0)
        if is_pushing > 0 and in_air == 0:
            step_reward += self.PEN_WALL_PUSH

        # --- MODULO 8: Power-Up e Sopravvivenza ---
        curr_shield = info.get('shield', 0)
        curr_invinc = info.get('invincible', 0)
        curr_shoes = info.get('shoes', 0)

        if curr_shield > self.prev_shield:
            step_reward += self.REW_POWERUP
            sparky_logger.log("🛡️ SCUDO ACQUISITO! +{r}", r=self.REW_POWERUP)
        if curr_invinc > self.prev_invincible:
            step_reward += self.REW_POWERUP
            sparky_logger.log("⭐ INVINCIBILITÀ! +{r}", r=self.REW_POWERUP)
        if curr_shoes > self.prev_shoes:
            step_reward += self.REW_POWERUP
            sparky_logger.log("👟 SPEED SHOES! +{r}", r=self.REW_POWERUP)

        self.prev_shield = curr_shield
        self.prev_invincible = curr_invinc
        self.prev_shoes = curr_shoes

        # --- MODULO 9: Annegamento (Labyrinth Zone) ---
        curr_air = info.get('air_timer', 1800)
        is_underwater = curr_y > info.get('water_level', 4000)

        if is_underwater:
            if curr_air < 720:  # Sotto i 12 secondi, panico indotto
                step_reward -= 0.5
            if curr_air > self.prev_air + 600:
                step_reward += self.REW_BREATH
                sparky_logger.log("🫧 BOLLA D'ARIA PRESA! +{r}", r=self.REW_BREATH)

        self.prev_air = curr_air

        # --- MODULO 10: Vite Extra (1-UP) ---
        curr_lives = info.get('lives', 3)
        if curr_lives > self.prev_lives:
            step_reward += self.REW_1UP
            sparky_logger.log("💚 VITA EXTRA (1-UP)! +{r}", r=self.REW_1UP)
        self.prev_lives = curr_lives

        # --- MODULO 11: Analisi Post-Mortem e Glitch ---
        if term:
            rings_lost_at_death = self.prev_rings
            ring_penalty = self.PEN_RING_MULT * rings_lost_at_death

            is_hazard_near = any(info.get(f'obj{i}_id', 0) in [54, 64, 34, 75, 84] for i in range(1, 17))
            is_falling = curr_y > info.get('y_limit_bottom', 4000)

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