import gymnasium as gym
from core.logger import sparky_logger


class SparkyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # --- CONFIGURAZIONE PREMI (PRUDENZA & EFFICIENZA) ---
        self.REW_TIME = -0.1  # Malus tempo
        self.REW_PROGRESS = 3.0  # Punti progresso
        self.PEN_BACKTRACK = -4.0  # Severo se torna indietro
        self.REW_WIN = 2500.0  # Obiettivo Supremo
        self.PEN_DEATH = -600.0  # Morire è il fallimento massimo

        self.REW_ENEMY = 250.0  # Premio nemico
        self.REW_RING = 15.0  # Premio anello
        self.REW_FIRST_RING = 100.0  # BONUS "SALVAVITA" (Passaggio 0 -> 1)
        self.PEN_VULNERABLE = -0.3  # Malus costante se ha 0 anelli
        self.PEN_DAMAGE = -200.0  # Malus colpo subito

        self.REW_SPRING = 400.0  # Trampolino
        self.REW_SPRING_RIGHT = 200.0  # Trampolino + Direzione Destra
        self.REW_LOOP_SUCCESS = 800.0  # Giro della morte completo
        self.PEN_JUMP_LOOP = -20.0  # Errore critico: saltare nel loop

        self.PEN_ACTION_FLICKER = -0.1  # Anti-azione frenetica (smoothness)
        self.PEN_WALL_STUCK = -0.8  # Pushing wall
        self.PEN_STUCK_IDLE = -5.0  # Malus se resta fermo nello stesso posto

        self.REW_BREATH = 500.0  # Bolla d'aria presa
        self.REW_CHECKPOINT = 500.0  # Palo stellare

        self._init_vars()

    def _init_vars(self):
        self.max_x = 0
        self.prev_x = 0
        self.prev_rings = 0
        self.prev_score = 0
        self.prev_action = -1
        self.prev_air = 1800
        self.prev_lamppost = 0
        self.stuck_counter = 0
        self.first_step = True

        # Stato Molla e Loop
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

        # Estrazione dati
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
        is_pushing = info.get('pushing_wall', 0) > 0
        underwater = (status & 64) != 0

        if self.first_step:
            self.max_x = self.prev_x = curr_x
            self.prev_rings = c_rings
            self.prev_score = curr_score
            self.prev_air = curr_air
            self.prev_lamppost = curr_lamppost
            self.first_step = False

        step_reward = self.REW_TIME

        # --- 1. MOVIMENTO E NAVIGAZIONE ---
        # Progresso e Backtrack
        if curr_x > self.max_x:
            step_reward += (curr_x - self.max_x) * self.REW_PROGRESS
            self.max_x = curr_x
        elif curr_x < self.prev_x - 1:
            step_reward += self.PEN_BACKTRACK

        # Stuck Idle (Se X non cambia per troppo tempo)
        if abs(curr_x - self.prev_x) < 2:
            self.stuck_counter += 1
            if self.stuck_counter > 120:  # 2 secondi di fermo
                step_reward += self.PEN_STUCK_IDLE
        else:
            self.stuck_counter = 0

        # Pushing Wall
        if is_pushing and not in_air:
            step_reward += self.PEN_WALL_STUCK

        # Anti-Azione Frenetica (Se cambia tasto ad ogni frame)
        if action != self.prev_action:
            step_reward += self.PEN_ACTION_FLICKER
        self.prev_action = action

        # --- 2. SOPRAVVIVENZA ---
        # Economia Anelli e Vulnerabilità
        if c_rings == 0:
            step_reward += self.PEN_VULNERABLE
        elif c_rings > self.prev_rings:
            if self.prev_rings == 0:
                step_reward += self.REW_FIRST_RING  # Bonus Salvavita!
                sparky_logger.log("🛡️ SALVAVITA! Preso primo anello.")
            else:
                step_reward += self.REW_RING * (c_rings - self.prev_rings)
        elif c_rings < self.prev_rings and not term:
            step_reward += self.PEN_DAMAGE
        self.prev_rings = c_rings

        # Gestione Aria (LZ)
        if underwater:
            if curr_air < 600:  # Meno di 10 secondi
                step_reward -= 1.0  # Panico
            if curr_air > self.prev_air + 500:  # Ha preso una bolla
                step_reward += self.REW_BREATH
                sparky_logger.log("🫧 ARIA! Presa bolla d'aria.")
        self.prev_air = curr_air

        # --- 3. COMBATTIMENTO E INTERAZIONE ---
        # Nemici (Counter accurato)
        score_gain = curr_score - self.prev_score
        if score_gain >= 100:
            num_enemies = score_gain // 100
            step_reward += self.REW_ENEMY * num_enemies
            sparky_logger.log("👾 NEMICO! +{r}", r=self.REW_ENEMY * num_enemies)
        self.prev_score = curr_score

        # Checkpoints
        if curr_lamppost > self.prev_lamppost:
            step_reward += self.REW_CHECKPOINT
            self.prev_lamppost = curr_lamppost
            sparky_logger.log("🚩 CHECKPOINT!")

        # Trampolini (Con Bonus Direzione Destra)
        radar_objects = info.get('ai_radar_slots', [])
        spring_nearby = any(obj['id'] == 65 for obj in radar_objects if obj['dist'] < 45)
        if spring_nearby and v_y < -800 and not self.spring_active:
            bonus = self.REW_SPRING
            if v_x > 150: bonus += self.REW_SPRING_RIGHT  # Incentivo andare a destra
            step_reward += bonus
            self.spring_active = True
            sparky_logger.log("🚀 TRAMPOLINO! +{r}", r=bonus)
        if not in_air: self.spring_active = False

        # --- 4. MECCANICHE AVANZATE (Giro della Morte) ---
        is_jump = action in [3, 4, 5]
        if 15 < angle < 240:
            self.loop_potential = True
            if 100 < angle < 150 and g_speed > 400:  # Velocità minima per restare appesi
                self.reached_top = True
            if is_jump:
                step_reward += self.PEN_JUMP_LOOP  # Punizione se salta nel loop
        else:
            if self.loop_potential and self.reached_top:
                if curr_x > self.max_x:  # Deve essere uscito andando avanti
                    step_reward += self.REW_LOOP_SUCCESS
                    sparky_logger.log("🌀 LOOP COMPLETATO! +{r}", r=self.REW_LOOP_SUCCESS)
                self.loop_potential = self.reached_top = False

        # --- 5. TERMINAZIONE ---
        if info.get('level_end_bonus', 0) > 0:
            step_reward += self.REW_WIN
            trunc = True
            sparky_logger.log("🏁 VITTORIA!")

        if term:
            step_reward += self.PEN_DEATH
            sparky_logger.log("💀 MORTE! Fine episodio.")

        return obs, step_reward / 100.0, term, trunc, info