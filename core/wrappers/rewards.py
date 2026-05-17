import gymnasium as gym
from core.logger import sparky_logger


class SparkyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # --- 1. CORE PROGRESS & FLUIDITA' ---
        self.REW_TIME = -0.8
        self.REW_PROGRESS = 5.0
        self.REW_WIN = 5000.0
        self.PEN_BACKTRACK = -5.0
        self.PEN_ACTION_FLICKER = -0.3
        self.REW_MOMENTUM_BASE = 5.0  # Mantenuta per la Grande Velocità

        # --- 2. SURVIVAL & COMBAT ---
        self.PEN_DEATH = -600.0  # Penalità assoluta se perde una VITA
        self.PEN_DAMAGE = -70.0  # Penalità se perde anelli (ma sopravvive)
        self.PEN_VULNERABLE = -0.5  # MALUS COSTANTE se ha 0 ANELLI (Incentiva la raccolta)
        self.REW_FIRST_RING = 50.0

        self.REW_ENEMY = 30.0
        self.REW_BOSS_HIT = 500.0
        self.REW_BOSS_PROXIMITY = 0.5
        self.REW_BOSS_DEFEAT = 10000.0

        # --- 3. ESPLORAZIONE, LOOP & VITAL ---
        self.REW_CHECKPOINT = 1000.0
        self.REW_SPRING = 50.0
        self.REW_POWERUP = 150.0  # Aggiunta variabile mancante!
        self.PEN_STUCK_WARNING = -10.0
        self.PEN_STUCK_FATAL = -300.0
        self.PEN_IDLE = -2.0
        self.PEN_JUMP_LOOP = -15.0
        self.REW_LOOP_SUCCESS = 1500.0

        self.BOSS_IDS = [61, 90, 117, 118, 121, 126]
        self._init_vars()

    def _init_vars(self):
        self.max_x = 0
        self.prev_x = 0
        self.prev_y = 0
        self.prev_rings = 0
        self.prev_score = 0
        self.prev_lamppost = 0
        self.prev_boss_hp = 8
        self.prev_action = 0
        self.prev_lives = 3
        self.stuck_counter = 0
        self.stuck_anchor_x = 0
        self.boss_initialized = False
        self.first_step = True
        self.prev_invincible = 0
        self.prev_shield = 0
        self.prev_shoes = 0
        self.loop_potential = False
        self.reached_top = False
        self.spring_active = False

    def reset(self, **kwargs):
        self._init_vars()
        return self.env.reset(**kwargs)

    def set_logger_state(self, state: bool):
        sparky_logger.console_enabled = state

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        # Estrazione Dati
        curr_x = info.get('x', 0)
        curr_y = info.get('y', 0)
        v_x = info.get('velocity_x', 0)
        v_y = info.get('velocity_y', 0)
        g_speed = info.get('ground_speed', 0)  # Nuova
        angle = info.get('angle', 0)  # Nuova
        c_rings = info.get('rings', 0)
        curr_score = info.get('score', 0)
        curr_lamppost = info.get('lamppost_id', 0)
        curr_lives = info.get('lives', 3)  # Estrazione vite

        ai_vec = info.get('ai_input_vector', [])

        curr_boss_hp = ai_vec[14] * 8.0 if len(ai_vec) > 14 else 0.0
        boss_dx = ai_vec[15] if len(ai_vec) > 15 else 0.0
        boss_dy = ai_vec[16] if len(ai_vec) > 16 else 0.0

        status = int(info.get('status', 0))
        in_air = (status & 2) != 0
        is_jump_action = action in [3, 4, 5, 7]

        curr_invinc = info.get('invincible', 0)
        curr_shield = info.get('shield', 0)
        curr_shoes = info.get('shoes', 0)

        # Sincronizzazione Primo Frame
        if self.first_step:
            self.max_x = self.prev_x = self.stuck_anchor_x = curr_x
            self.prev_y = curr_y
            self.prev_rings = c_rings
            self.prev_score = curr_score
            self.prev_lamppost = curr_lamppost
            self.prev_boss_hp = curr_boss_hp
            self.prev_invincible = curr_invinc
            self.prev_shield = curr_shield
            self.prev_shoes = curr_shoes
            self.prev_action = action
            self.prev_lives = curr_lives
            self.first_step = False
            return obs, 0.0, term, trunc, info

        is_boss_active = any(info.get(f'obj{i}_id', 0) in self.BOSS_IDS for i in range(1, 61))

        # Inizio calcolo reward dello step
        step_reward = self.REW_TIME

        # --- 0. ANTI-FLICKER (SMOOTH MOVEMENT) ---
        if action != self.prev_action:
            step_reward += self.PEN_ACTION_FLICKER
        self.prev_action = action

        # --- 1. MODULO MOVIMENTO, LOOP & FISICA ---
        if curr_x > self.max_x:
            step_reward += (curr_x - self.max_x) * self.REW_PROGRESS
            self.max_x = curr_x
        elif curr_x < self.prev_x - 1:
            if not is_boss_active:  # Nessuna penalità se sta combattendo il boss!
                step_reward += self.PEN_BACKTRACK

        if not in_air:
            self.spring_active = False
            # LOGICA LOOP DELLA MORTE
            # Angolo tra 45 e 215 (scala Sonic 0-255: ~32 a ~150)
            if 32 < angle < 150:
                self.loop_potential = True
                if 70 < angle < 110 and g_speed > 450:  # È quasi a testa in giù con buona velocità
                    self.reached_top = True

                if is_jump_action:
                    step_reward += self.PEN_JUMP_LOOP
                    sparky_logger.log("❌ SALTO NEL LOOP!")
            else:
                if self.loop_potential and self.reached_top:
                    if curr_x > self.max_x:  # È uscito dal loop in avanti
                        step_reward += self.REW_LOOP_SUCCESS
                        sparky_logger.log("🌀 LOOP COMPLETATO! +{r}", r=self.REW_LOOP_SUCCESS)
                    self.loop_potential = self.reached_top = False

            # INERZIA ESPONENZIALE (Grande Velocità a terra)
            if g_speed > 1400 and (curr_x >= self.max_x - 50):
                step_reward += (g_speed / 1000.0) ** 2 * self.REW_MOMENTUM_BASE
                # Commentato il logger per non spammare la console a 60 frame al secondo
                #sparky_logger.log("👟 GRANDE VELOCITA' +{s}", s= momentum_bonus)

            if abs(v_x) < 50:
                step_reward += self.PEN_IDLE
                #sparky_logger.log("🐢 FERMO/COVARDO! Penalità in corso.")

        # LOGICA TRAMPOLINI (Basata puramente sulla fisica)
        if v_y <= -2200 and not self.spring_active:
            step_reward += self.REW_SPRING
            self.spring_active = True
            sparky_logger.log("🚀 TRAMPOLINO UTILIZZATO! +{r}", r=self.REW_SPRING)

        # ANTI-STUCK
        if is_boss_active or abs(curr_x - self.stuck_anchor_x) > 30:
            self.stuck_counter = 0
            self.stuck_anchor_x = curr_x
        else:
            self.stuck_counter += 1
            if self.stuck_counter == 120:
                step_reward += self.PEN_STUCK_WARNING
                sparky_logger.log("⚠️ STUCK WARNING! (2s)")
            elif self.stuck_counter >= 360:
                step_reward += self.PEN_STUCK_FATAL
                trunc = True
                sparky_logger.log("🛑 FATAL STUCK! Reset episodio.")

        # --- 2. MODULO SOPRAVVIVENZA & POWER-UPS ---

        # LOGICA MORTE VS DANNO (Correggendo il problema delle vite)
        # Se le vite diminuiscono, è matematicamente morto (vuoto o no anelli)
        if curr_lives < self.prev_lives:
            step_reward += self.PEN_DEATH
            sparky_logger.log("💀 VITA PERSA! (Morte rilevata) -{p}", p=abs(self.PEN_DEATH))
        else:
            # Se è vivo, controlliamo se ha perso anelli (danno leggero)
            if c_rings < self.prev_rings:
                step_reward += self.PEN_DAMAGE
                sparky_logger.log("💥 DANNO SUBITO (Anelli Persi)! -{p}", p=abs(self.PEN_DAMAGE))
            elif c_rings > self.prev_rings:
                # Premio per la raccolta anelli
                if self.prev_rings == 0:
                    step_reward += self.REW_FIRST_RING
                    sparky_logger.log("🛡️ ARMATURA ATTIVA (Primo Anello)! +{s}", s=self.REW_FIRST_RING)
                else:
                    step_reward += (c_rings - self.prev_rings) * 1.0  # 1 punto per anello extra

        # Power-Ups
        if curr_invinc > self.prev_invincible:
            step_reward += self.REW_POWERUP
            sparky_logger.log("⭐ INVINCIBILITA' PRESA!")
        if curr_shield > self.prev_shield:
            step_reward += self.REW_POWERUP
            sparky_logger.log("🛡️ SCUDO PRESO!")
        if curr_shoes > self.prev_shoes:
            step_reward += self.REW_POWERUP
            sparky_logger.log("👟 SCARPE PRESE!")

        # --- 3. MODULO COMBATTIMENTO ---
        # Nemici (Badniks)
        score_gain = curr_score - self.prev_score
        if score_gain >= 100:
            step_reward += self.REW_ENEMY
            sparky_logger.log("👾 NEMICO SCONFITTO! +{r}", r=self.REW_ENEMY)

        if is_boss_active:
            if not self.boss_initialized:
                self.prev_boss_hp = curr_boss_hp
                self.boss_initialized = True
            else:
                # Distanza euclidea dal Boss usando i dati del Radar
                dist_from_boss = (boss_dx ** 2 + boss_dy ** 2) ** 0.5

                # Se è molto vicino al boss (sotto di lui o sta per colpirlo), riceve punti costanti
                if dist_from_boss < 0.25:
                    step_reward += self.REW_BOSS_PROXIMITY

                # Boss Colpito
                if curr_boss_hp < self.prev_boss_hp:
                    step_reward += self.REW_BOSS_HIT
                    sparky_logger.log("🎯 BOSS COLPITO! +{r}", r=self.REW_BOSS_HIT)

                    # Boss Sconfitto! (Scende a 0)
                    if curr_boss_hp <= 0:
                        step_reward += self.REW_BOSS_DEFEAT
                        sparky_logger.log("👑 BOSS SCONFITTO!!! +{r}", r=self.REW_BOSS_DEFEAT)

                self.prev_boss_hp = curr_boss_hp

        else:
            self.boss_initialized = False

            # --- 4. MODULO INTERAZIONE & TERMINAZIONE ---
        if curr_lamppost > self.prev_lamppost:
            step_reward += self.REW_CHECKPOINT
            sparky_logger.log("🚩 CHECKPOINT RAGGIUNTO! +{r}", r=self.REW_CHECKPOINT)

        if info.get('level_end_bonus', 0) > 0:
            step_reward += self.REW_WIN
            trunc = True
            sparky_logger.log("🏁 VITTORIA LIVELLO!")

        if term:  # Game Over totale (0 vite)
            step_reward += self.PEN_DEATH

        # Aggiornamento variabili
        self.prev_x = curr_x
        self.prev_y = curr_y
        self.prev_rings = c_rings
        self.prev_score = curr_score
        self.prev_lamppost = curr_lamppost
        self.prev_invincible = curr_invinc
        self.prev_shield = curr_shield
        self.prev_shoes = curr_shoes
        self.prev_lives = curr_lives

        # Divisione finale per scalare i valori verso PPO
        return obs, step_reward / 100.0, term, trunc, info