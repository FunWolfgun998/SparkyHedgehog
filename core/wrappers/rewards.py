import gymnasium as gym
from core.logger import sparky_logger


class SparkyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # --- 1. CORE PROGRESS (IL MOTORE) ---
        self.REW_TIME = -0.2  # Piccolo malus costante per incoraggiare il movimento
        self.REW_PROGRESS = 2.0  # Punti per ogni pixel guadagnato verso destra
        self.REW_WIN = 5000.0  # Obiettivo Supremo (Win Condition)
        self.PEN_DEATH = -500.0  # Malus per la morte (da evitare assolutamente)

        # --- 2. SURVIVAL & COMBAT (LA SOPRAVVIVENZA) ---
        self.REW_FIRST_RING = 50.0  # Passaggio critico da 0 a 1 anello (Armatura ON)
        self.REW_RING = 1.0  # Anelli extra (valore minimo per evitare farming)
        self.PEN_DAMAGE = -100.0  # Malus per perdita anelli (Armatura OFF)
        self.REW_ENEMY = 30.0  # Piccolo premio per nemici (non prioritario)
        self.REW_BOSS_HIT = 500.0  # Premio per colpire il boss (necessario per avanzare)

        # --- 3. ESPLORAZIONE & VITAL ---
        self.REW_CHECKPOINT = 1000.0  # Grande incentivo a raggiungere i pali stellari
        self.REW_BREATH = 100.0  # Premio per bolle d'aria (sopravvivenza in acqua)
        self.REW_POWERUP = 150.0
        self.PEN_STUCK_WARNING = -10.0
        self.PEN_STUCK_FATAL = -50.0
        self.REW_MOMENTUM_BASE = 5.0

        self.BOSS_IDS = [61, 90, 117, 118, 121, 126]
        self._init_vars()

    def _init_vars(self):
        self.max_x = 0
        self.prev_x = 0
        self.prev_y = 0
        self.prev_rings = 0
        self.prev_score = 0
        self.prev_air = 1800
        self.prev_lamppost = 0
        self.prev_boss_hp = 8
        self.stuck_counter = 0
        self.stuck_anchor_x = 0
        self.boss_initialized = False
        self.first_step = True
        self.prev_invincible = 0
        self.prev_shield = 0
        self.prev_shoes = 0
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
        c_rings = info.get('rings', 0)
        curr_score = info.get('score', 0)
        curr_air = info.get('air_timer', 1800)
        curr_lamppost = info.get('lamppost_id', 0)
        curr_boss_hp = info.get('boss_hp', 0)

        status = int(info.get('status', 0))
        in_air = (status & 2) != 0
        underwater = (status & 64) != 0

        curr_invinc = info.get('invincible', 0)
        curr_shield = info.get('shield', 0)
        curr_shoes = info.get('shoes', 0)

        # Sincronizzazione Primo Frame
        if self.first_step:
            self.max_x = self.prev_x = self.stuck_anchor_x = curr_x
            self.prev_y = curr_y
            self.prev_rings = c_rings
            self.prev_score = curr_score
            self.prev_air = curr_air
            self.prev_lamppost = curr_lamppost
            self.prev_boss_hp = curr_boss_hp
            self.prev_invincible = curr_invinc
            self.prev_shield = curr_shield
            self.prev_shoes = curr_shoes
            self.first_step = False
            return obs, 0.0, term, trunc, info

        # Inizio calcolo reward dello step
        step_reward = self.REW_TIME

        # --- 1. MODULO MOVIMENTO & FISICA AVANZATA ---
        if curr_x > self.max_x:
            step_reward += (curr_x - self.max_x) * self.REW_PROGRESS
            self.max_x = curr_x

        # INERZIA ESPONENZIALE (Impedisce saltelli inutili e fa superare i loop)
        if v_x > 300 and not in_air and (curr_x >= self.max_x - 50):
            #sparky_logger.log("👟 GRANDE VELOCITA'")
            speed_ratio = v_x / 1000.0
            momentum_bonus = (speed_ratio ** 2) * self.REW_MOMENTUM_BASE
            step_reward += momentum_bonus

        # --- 2. MODULO ANTI-STUCK ---
        is_boss_active = any(info.get(f'obj{i}_id', 0) in self.BOSS_IDS for i in range(1, 61))

        if is_boss_active or abs(curr_x - self.stuck_anchor_x) > 15:
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

        # --- 3. MODULO SOPRAVVIVENZA & POWER-UPS ---
        # Anelli
        if c_rings > self.prev_rings:
            if self.prev_rings == 0:
                step_reward += self.REW_FIRST_RING
                sparky_logger.log("🛡️ ARMATURA ATTIVA (Primo Anello)! +{s}", s=self.REW_FIRST_RING)
            else:
                step_reward += (c_rings - self.prev_rings) * self.REW_RING
        elif c_rings < self.prev_rings and not term:
            step_reward += self.PEN_DAMAGE
            sparky_logger.log("💥 DANNO SUBITO (Anelli Persi)! -{p}", p=abs(self.PEN_DAMAGE))

        # Bolla d'aria (Corretto il trigger)
        if underwater and (curr_air > self.prev_air + 200):
            step_reward += self.REW_BREATH
            sparky_logger.log("🫧 ARIA PRESA! +{r}", r=self.REW_BREATH)

        # Power-Ups (Ripristinati)
        if curr_invinc > self.prev_invincible:
            step_reward += self.REW_POWERUP
            sparky_logger.log("⭐ INVINCIBILITA' PRESA!")
        if curr_shield > self.prev_shield:
            step_reward += self.REW_POWERUP
            sparky_logger.log("🛡️ SCUDO PRESO!")
        if curr_shoes > self.prev_shoes:
            step_reward += self.REW_POWERUP
            sparky_logger.log("👟 SCARPE PRESE!")

        # --- MODULO COMBATTIMENTO ---
        # Nemici (Badniks)
        score_gain = curr_score - self.prev_score
        if score_gain >= 100:
            step_reward += self.REW_ENEMY
            sparky_logger.log("👾 NEMICO SCONFITTO! +{r}", r=self.REW_ENEMY)

        # Boss
        if is_boss_active:
            if not self.boss_initialized:
                self.prev_boss_hp = curr_boss_hp
                self.boss_initialized = True
            elif curr_boss_hp < self.prev_boss_hp:
                step_reward += self.REW_BOSS_HIT
                sparky_logger.log("🎯 BOSS COLPITO! +{r}", r=self.REW_BOSS_HIT)
                self.prev_boss_hp = curr_boss_hp
        else:
            self.boss_initialized = False

        # --- MODULO INTERAZIONE ---
        if curr_lamppost > self.prev_lamppost:
            step_reward += self.REW_CHECKPOINT
            sparky_logger.log("🚩 CHECKPOINT RAGGIUNTO! +{r}", r=self.REW_CHECKPOINT)

        # --- TERMINAZIONE ---
        if info.get('level_end_bonus', 0) > 0:
            step_reward += self.REW_WIN
            trunc = True
            sparky_logger.log("🏁 VITTORIA LIVELLO!")

        if term:  # Morte
            step_reward += self.PEN_DEATH
            sparky_logger.log("💀 MORTE!")

        # Aggiornamento variabili per il prossimo frame
        self.prev_x = curr_x
        self.prev_y = curr_y
        self.prev_rings = c_rings
        self.prev_score = curr_score
        self.prev_air = curr_air
        self.prev_lamppost = curr_lamppost
        self.prev_invincible= curr_invinc
        self.prev_shield = curr_shield
        self.prev_shoes = curr_shoes

        # Divisione finale per scalare i valori verso PPO
        return obs, step_reward / 100.0, term, trunc, info