import gymnasium as gym
from core.logger import sparky_logger


class SparkyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # --- 1. CORE PROGRESS (IL MOTORE) ---
        self.REW_TIME = -0.1  # Piccolo malus costante per incoraggiare il movimento
        self.REW_PROGRESS = 2.0  # Punti per ogni pixel guadagnato verso destra
        self.REW_WIN = 5000.0  # Obiettivo Supremo (Win Condition)
        self.PEN_DEATH = -500.0  # Malus per la morte (da evitare assolutamente)

        # --- 2. SURVIVAL & COMBAT (LA SOPRAVVIVENZA) ---
        self.REW_FIRST_RING = 50.0  # Passaggio critico da 0 a 1 anello (Armatura ON)
        self.REW_RING = 1.0  # Anelli extra (valore minimo per evitare farming)
        self.PEN_DAMAGE = -100.0  # Malus per perdita anelli (Armatura OFF)
        self.REW_ENEMY = 30.0  # Piccolo premio per nemici (non prioritario)
        self.REW_BOSS_HIT = 200.0  # Premio per colpire il boss (necessario per avanzare)

        # --- 3. ESPLORAZIONE & VITAL ---
        self.REW_CHECKPOINT = 1000.0  # Grande incentivo a raggiungere i pali stellari
        self.REW_BREATH = 100.0  # Premio per bolle d'aria (sopravvivenza in acqua)

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
        self.boss_initialized = False
        self.first_step = True

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
        c_rings = info.get('rings', 0)
        curr_score = info.get('score', 0)
        curr_air = info.get('air_timer', 1800)
        curr_lamppost = info.get('lamppost_id', 0)
        curr_boss_hp = info.get('boss_hp', 0)

        # Sincronizzazione Primo Frame
        if self.first_step:
            self.max_x = self.prev_x = curr_x
            self.prev_y = curr_y
            self.prev_rings = c_rings
            self.prev_score = curr_score
            self.prev_air = curr_air
            self.prev_lamppost = curr_lamppost
            self.prev_boss_hp = curr_boss_hp
            self.first_step = False
            return obs, 0.0, term, trunc, info

        # Inizio calcolo reward dello step
        step_reward = self.REW_TIME

        # --- MODULO MOVIMENTO (Semplificato) ---
        # Premiamo solo se superiamo il punto più a destra mai raggiunto
        if curr_x > self.max_x:
            step_reward += (curr_x - self.max_x) * self.REW_PROGRESS
            self.max_x = curr_x
        # Rimosso malus Backtrack: REW_TIME è sufficiente a scoraggiare perdite di tempo

        # --- MODULO ANTI-STUCK (Semplificato) ---
        # Se la posizione è identica (tolleranza 1px), incrementa contatore
        if abs(curr_x - self.prev_x) <= 1 and abs(curr_y - self.prev_y) <= 1:
            self.stuck_counter += 1
            if self.stuck_counter >= 360:  # 6 secondi a 60fps
                trunc = True
                sparky_logger.log("🛑 STUCK DETECTED! Reset episodio.")
        else:
            self.stuck_counter = 0

        # --- MODULO SOPRAVVIVENZA (Logica Binaria Anelli) ---
        if c_rings > self.prev_rings:
            if self.prev_rings == 0:
                step_reward += self.REW_FIRST_RING  # Bonus Armatura ON
                sparky_logger.log("🛡️ ARMATURA ATTIVA (Primo Anello)! +{s}", s=self.REW_FIRST_RING)
            else:
                step_reward += (c_rings - self.prev_rings) * self.REW_RING
        elif c_rings < self.prev_rings and not term:
            step_reward += self.PEN_DAMAGE  # Malus Armatura OFF
            sparky_logger.log("💥 DANNO SUBITO (Anelli Persi)! -{p}", p=abs(self.PEN_DAMAGE))

        # Bolla d'aria
        if curr_air > self.prev_air + 500:
            step_reward += self.REW_BREATH
            sparky_logger.log("🫧 ARIA PRESA!")

        # --- MODULO COMBATTIMENTO ---
        # Nemici (Badniks)
        score_gain = curr_score - self.prev_score
        if score_gain >= 100:
            step_reward += self.REW_ENEMY
            sparky_logger.log("👾 NEMICO SCONFITTO! +{r}", r=self.REW_ENEMY)

        # Boss
        is_boss_active = any(info.get(f'obj{i}_id', 0) in self.BOSS_IDS for i in range(1, 61))
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

        # Divisione finale per scalare i valori verso PPO
        return obs, step_reward / 100.0, term, trunc, info