import os
import cv2
import numpy as np
import math
from stable_baselines3.common.callbacks import BaseCallback
from core.logger import sparky_logger
from core.hud import SparkyHUD
import config


class SparkyRoundCheckpoint(BaseCallback):
    def __init__(self, save_freq, save_path, run_number, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.run_number = run_number

    def _on_step(self) -> bool:
        total_steps = self.model.num_timesteps
        offset = config.NUM_ENVS
        if (total_steps // self.save_freq) > ((total_steps - offset) // self.save_freq):
            rounded = (total_steps // self.save_freq) * self.save_freq
            fname = f"Sparky_run_{self.run_number}_{rounded}.zip"
            save_loc = os.path.join(self.save_path, fname)
            self.model.save(save_loc)
            if self.verbose > 0:
                print(f"\n💾 [CHECKPOINT] Modello salvato a {rounded} passi: {fname}")
        return True


class ShallyTurboCallback(BaseCallback):
    def __init__(self, render_freq=30, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.show_vision = False
        self.smooth_mode = False
        self.logs_on = True  # Stato iniziale dei log
        self.hud = SparkyHUD()

        self.ctrl_img = np.zeros((200, 320, 3), dtype=np.uint8)
        cv2.putText(self.ctrl_img, "V = ON/OFF Griglia", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(self.ctrl_img, "S = Fluido (LENTO)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(self.ctrl_img, "H = Toggle HUD", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(self.ctrl_img, "P = Toggle Logs", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

    def _on_step(self):
        cv2.imshow("CONTROLLO", self.ctrl_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('v'):
            self.show_vision = not self.show_vision
            if not self.show_vision:
                try:
                    cv2.destroyWindow("Multi-Sonic Monitor")
                except:
                    pass

        if key == ord('s'):
            # Se attivo il fluido, attivo anche la visione automaticamente
            self.smooth_mode = not self.smooth_mode
            if self.smooth_mode: self.show_vision = True

        if key == ord('h'):
            self.hud.toggle()

        if key == ord('p'):
            self.logs_on = not self.logs_on
            # Comandiamo a tutti gli ambienti di cambiare lo stato del loro logger
            self.training_env.env_method("set_logger_state", self.logs_on)

            stato = "ATTIVI" if self.logs_on else "DISABILITATI (Performance Mode 🚀)"
            print(f"\n⚙️ [COMMAND CENTER] Log a schermo: {stato}\n", flush=True)

        if self.show_vision:
            freq = 1 if self.smooth_mode else self.render_freq
            if self.n_calls % freq == 0:
                imgs = self.training_env.get_images()
                infos = self.locals.get('infos', [])  # Recupera le info per l'HUD

                if imgs is not None and len(imgs) > 0:
                    num_envs = len(imgs)
                    cols = math.ceil(math.sqrt(num_envs))
                    rows = math.ceil(num_envs / cols)

                    # Definiamo la dimensione di ogni singolo Sonic nella griglia
                    # Se l'HUD è attivo usiamo la scala del HUD, altrimenti originale
                    m_w, m_h = 320, 224

                    grid_img = np.zeros((rows * m_h, cols * m_w, 3), dtype=np.uint8)

                    for i, img in enumerate(imgs):
                        current_info = infos[i] if i < len(infos) else {}

                        # 1. Applica HUD se abilitato (restituisce il frame elaborato)
                        # Nota: hud.overlay si occupa di fare il resize interno
                        processed = self.hud.overlay(img, current_info)

                        # 2. Converte in BGR per OpenCV
                        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

                        # 3. Lo rimpicciolisce per farlo stare nella griglia da 30
                        # (Altrimenti la finestra sarebbe di 5000 pixel)
                        sf = cv2.resize(processed_bgr, (m_w, m_h), interpolation=cv2.INTER_AREA)

                        r, c = i // cols, i % cols
                        grid_img[r * m_h:(r + 1) * m_h, c * m_w:(c + 1) * m_w] = sf

                    cv2.imshow("Multi-Sonic Monitor", grid_img)
        else:
            cv2.destroyWindow("Multi-Sonic Monitor")
        return True