import gymnasium as gym
from core.logger import sparky_logger


class SparkyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # --- CONFIGURAZIONE PREMI (PRUDENZA & EFFICIENZA) ---
        self.REW_TIME = -0.2  # Malus tempo
        self.REW_PROGRESS = 3.0  # Punti progresso
        self.PEN_BACKTRACK = -2.0  # Severo se torna indietro
        self.REW_WIN = 1250.0  # Obiettivo Supremo
        self.PEN_DEATH = -200.0  # Morire è il fallimento massimo

        self.REW_ENEMY = 250.0  # Premio nemico
        self.REW_RING = 15.0  # Premio anello
        self.REW_FIRST_RING = 30.0  # BONUS "SALVAVITA" (Passaggio 0 -> 1)
        self.PEN_VULNERABLE = -0.3  # Malus costante se ha 0 anelli
        self.PEN_DAMAGE = -80.0  # Malus colpo subito

        self.REW_POWERUP = 300.0  # Premio per Scudo, Stelle o Scarpe
        self.REW_SPRING = 400.0  # Trampolino
        self.REW_SPRING_RIGHT = 100.0  # Trampolino + Direzione Destra
        self.REW_LOOP_SUCCESS = 800.0  # Giro della morte completo
        self.PEN_JUMP_LOOP = -1.0  # Errore critico: saltare nel loop

        self.PEN_ACTION_FLICKER = -0.4  # Anti-azione frenetica (smoothness)
        self.PEN_WALL_STUCK = -0.8  # Pushing wall

        # --- NUOVE VARIABILI PER LOGICA STUCK ---
        self.PEN_STUCK_WARNING = -25.0  # Malus una tantum a 2 secondi
        self.PEN_STUCK_FATAL = -50.0  # Malus prima del ricaricamento a 6 secondi
        self.BOSS_IDS = [61, 90, 117, 118, 121, 126]

        self.REW_BREATH = 500.0  # Bolla d'aria presa
        self.REW_CHECKPOINT = 500.0  # Palo stellare
        self.REW_BOSS_HIT = 500.0  # Colpo inflitto al Boss

        self.REW_LIFE_GAIN = 1000.0  # Vita Extra (1-UP o 100 anelli)
        self.REW_ROLLING = 0.5  # Incoraggiamento difesa (Rotolamento)
        self.REW_PROGRESS_Y = 2.0  # Progresso verticale (Salita)
        self._init_vars()

    def _init_vars(self):
        self.max_x = 0
        self.prev_x = 0
        self.prev_y = 0
        self.min_y = 0
        self.prev_rings = 0
        self.prev_score = 0
        self.prev_action = -1
        self.prev_air = 1800
        self.prev_lamppost = 0
        self.prev_v_y = 0
        self.stuck_counter = 0
        self.prev_invincible = 0
        self.prev_shield = 0
        self.prev_shoes = 0
        self.prev_lives = 3
        self.prev_boss_hp = 8
        self.boss_initialized = False
        self.first_step = True
        self.spring_active = False
        self.loop_potential = False
        self.reached_top = False

    def reset(self, **kwargs):
        self._init_vars()
        return self.env.reset(**kwargs)

    def set_logger_state(self, state: bool):
        sparky_logger.console_enabled = state

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        # 1. Estrazione Dati
        curr_x = info.get('x', 0)
        curr_y = info.get('y', 0)
        v_x, v_y = info.get('velocity_x', 0), info.get('velocity_y', 0)
        g_speed = abs(info.get('ground_speed', 0))
        angle = info.get('angle', 0)
        c_rings = info.get('rings', 0)
        curr_score = info.get('score', 0)
        curr_air = info.get('air_timer', 1800)
        curr_lamppost = info.get('lamppost_id', 0)
        status = int(info.get('status', 0))
        in_air = (status & 2) != 0
        is_rolling = (status & 4) != 0
        underwater = (status & 64) != 0
        curr_boss_hp = info.get('boss_hp', 0)
        curr_lives = info.get('lives', 3)
        curr_zone = info.get('zone', 0)

        # 2. Sincronizzazione Primo Frame
        if self.first_step:
            self.max_x = self.prev_x = curr_x
            self.prev_y = self.min_y = curr_y
            self.prev_rings = c_rings
            self.prev_score = curr_score
            self.prev_air = curr_air
            self.prev_lamppost = curr_lamppost
            self.prev_lives = curr_lives
            self.prev_boss_hp = curr_boss_hp
            self.prev_v_y = v_y
            self.first_step = False
            return obs, 0.0, term, trunc, info

        step_reward = self.REW_TIME

        # --- MODULO MOVIMENTO ---
        if curr_x > self.max_x:
            step_reward += (curr_x - self.max_x) * self.REW_PROGRESS
            self.max_x = curr_x
        elif curr_x < self.prev_x - 1:
            step_reward += self.PEN_BACKTRACK

        if curr_zone in [1, 2, 4] and curr_y < self.min_y:
            step_reward += (self.min_y - curr_y) * self.REW_PROGRESS_Y
            self.min_y = curr_y

        # --- MODULO ANTI-STUCK ---
        is_boss_active = any(info.get(f'obj{i}_id', 0) in self.BOSS_IDS for i in range(1, 61))
        if not is_boss_active:
            if abs(curr_x - self.prev_x) <= 2 and abs(curr_y - self.prev_y) <= 2:
                self.stuck_counter += 1
                if self.stuck_counter == 120:
                    step_reward += self.PEN_STUCK_WARNING
                    sparky_logger.log("⚠️ STUCK WARNING! (2s)")
                elif self.stuck_counter >= 360:
                    step_reward += self.PEN_STUCK_FATAL
                    trunc = True
                    sparky_logger.log("🛑 FATAL STUCK! Reset.")
            else:
                self.stuck_counter = 0
        else:
            self.stuck_counter = 0

        if info.get('pushing_wall', 0) > 0 and not in_air:
            step_reward += self.PEN_WALL_STUCK

        if action != self.prev_action:
            step_reward += self.PEN_ACTION_FLICKER
        self.prev_action = action

        # --- MODULO SOPRAVVIVENZA ---
        if c_rings > self.prev_rings:
            if self.prev_rings == 0:
                step_reward += self.REW_FIRST_RING
                sparky_logger.log("🛡️ SALVAVITA! +{s}", s=self.REW_FIRST_RING)
            else:
                step_reward += self.REW_RING * (c_rings - self.prev_rings)
        elif c_rings < self.prev_rings and not term:
            step_reward += self.PEN_DAMAGE
            sparky_logger.log("💥 DANNO! -{p}", p=self.PEN_DAMAGE)

        if c_rings == 0: step_reward += self.PEN_VULNERABLE
        self.prev_rings = c_rings

        if underwater:
            if curr_air < 600: step_reward -= 1.0
            if curr_air > self.prev_air + 500:
                step_reward += self.REW_BREATH
                sparky_logger.log("🫧 ARIA!")
        self.prev_air = curr_air

        if curr_lives > self.prev_lives:
            step_reward += self.REW_LIFE_GAIN
            sparky_logger.log("💚 VITA EXTRA! +{r}", r=self.REW_LIFE_GAIN)
        self.prev_lives = curr_lives

        # --- MODULO COMBATTIMENTO ---
        score_gain = curr_score - self.prev_score
        if score_gain >= 100:
            step_reward += self.REW_ENEMY * (score_gain // 100)
            sparky_logger.log("👾 NEMICO! +{r}", r=self.REW_ENEMY)
        self.prev_score = curr_score

        if is_boss_active:
            if not self.boss_initialized:
                self.prev_boss_hp = curr_boss_hp
                self.boss_initialized = True
            elif curr_boss_hp < self.prev_boss_hp:
                step_reward += self.REW_BOSS_HIT
                sparky_logger.log("🎯 BOSS HIT! +{r}", r=self.REW_BOSS_HIT)
                self.prev_boss_hp = curr_boss_hp
        else:
            self.boss_initialized = False

        if is_rolling:
            radar_slots = info.get('ai_radar_slots', [])
            if any(obj['e'] == 1 and obj['dist'] < 100 for obj in radar_slots):
                step_reward += self.REW_ROLLING

        # --- MODULO INTERAZIONE ---
        if curr_lamppost > self.prev_lamppost:
            step_reward += self.REW_CHECKPOINT
            self.prev_lamppost = curr_lamppost
            sparky_logger.log("🚩 CHECKPOINT!")

        # Logica Trampolini (Indipendente)
        delta_v_y = v_y - self.prev_v_y
        radar_slots = info.get('ai_radar_slots', [])
        spring_hit = any(obj['id'] in [41, 65, 66] and obj['dy'] > 0 and abs(obj['dx']) < 0.04
                         for obj in radar_slots if obj['dist'] < 20)

        if spring_hit and delta_v_y < -400 and not self.spring_active:
            bonus = self.REW_SPRING
            if v_x > 100: bonus += self.REW_SPRING_RIGHT
            step_reward += bonus
            self.spring_active = True
            sparky_logger.log("🚀 TRAMPOLINO VERO! +{r}", r=bonus)

        if not in_air: self.spring_active = False
        self.prev_v_y = v_y

        # --- MODULO POWER-UP ---
        curr_invinc = info.get('invincible', 0)
        curr_shield = info.get('shield', 0)
        curr_shoes = info.get('shoes', 0)

        if curr_invinc > self.prev_invincible:
            step_reward += self.REW_POWERUP
            sparky_logger.log("⭐ INVINCIBILITA'!")
        if curr_shield > self.prev_shield:
            step_reward += self.REW_POWERUP
            sparky_logger.log("🛡️ SCUDO!")
        if curr_shoes > self.prev_shoes:
            step_reward += self.REW_POWERUP
            sparky_logger.log("👟 SCARPE!")

        self.prev_invincible, self.prev_shield, self.prev_shoes = curr_invinc, curr_shield, curr_shoes

        # --- MODULO LOOP ---
        if 20 < angle < 240:
            self.loop_potential = True
            if 100 < angle < 150 and g_speed > 350: self.reached_top = True
            if action in [3, 4, 5]:
                step_reward += self.PEN_JUMP_LOOP
                # sparky_logger.log("❌ ERRORE LOOP!")
        else:
            if self.loop_potential and self.reached_top:
                if curr_x > self.max_x:
                    step_reward += self.REW_LOOP_SUCCESS
                    sparky_logger.log("🌀 LOOP COMPLETATO! +{r}", r=self.REW_LOOP_SUCCESS)
                self.loop_potential = self.reached_top = False

        # --- TERMINAZIONE ---
        if info.get('level_end_bonus', 0) > 0:
            step_reward += self.REW_WIN
            trunc = True
            sparky_logger.log("🏁 VITTORIA!")
        if term:
            step_reward += self.PEN_DEATH
            sparky_logger.log("💀 MORTE!")

        self.prev_x, self.prev_y = curr_x, curr_y
        return obs, step_reward / 100.0, term, trunc, info