import os
import cv2
import numpy as np
import math
from stable_baselines3.common.callbacks import BaseCallback
from core.logger import sparky_logger

class SparkyRoundCheckpoint(BaseCallback):
    """Salva il modello ogni milione di passi globali in modo preciso."""

    def __init__(self, save_freq, save_path, run_number, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.run_number = run_number

    def _on_step(self) -> bool:
        # Usiamo il conteggio globale dei passi (num_timesteps)
        total_steps = self.model.num_timesteps

        # Calcoliamo se siamo passati sopra un multiplo di 'save_freq'
        # Usiamo config.NUM_ENVS come offset per catturare il momento esatto in sistemi paralleli
        if (total_steps // self.save_freq) > ((total_steps - 30) // self.save_freq):
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

        self.ctrl_img = np.zeros((160, 300, 3), dtype=np.uint8)
        cv2.putText(self.ctrl_img, "V = Grid (Veloce)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(self.ctrl_img, "S = Fluido (LENTO!)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(self.ctrl_img, "P = Toggle Logs", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

    def _on_step(self):
        cv2.imshow("CONTROLLO", self.ctrl_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('v'):
            self.show_vision, self.smooth_mode = not self.show_vision , False
        elif key == ord('s'):
            self.smooth_mode, self.show_vision = not self.smooth_mode, not self.smooth_mode
        elif key == ord('p'): #print management
            sparky_logger.toggle_performance_mode()

        if self.show_vision:
            freq = 1 if self.smooth_mode else self.render_freq
            if self.n_calls % freq == 0:
                imgs = self.training_env.get_images()
                if imgs is not None and len(imgs) > 0:
                    num_envs = len(imgs)
                    cols = math.ceil(math.sqrt(num_envs))
                    rows = math.ceil(num_envs / cols)
                    m_width, m_height = 320, 224
                    grid_img = np.zeros((rows * m_height, cols * m_width, 3), dtype=np.uint8)
                    for i, img in enumerate(imgs):
                        sf = cv2.resize(img, (m_width, m_height), interpolation=cv2.INTER_NEAREST)
                        sf = cv2.cvtColor(sf, cv2.COLOR_RGB2BGR)
                        r, c = i // cols, i % cols
                        grid_img[r * m_height:(r + 1) * m_height, c * m_width:(c + 1) * m_width] = sf
                    cv2.imshow("Multi-Sonic Monitor", grid_img)
        else:
            cv2.destroyWindow("Multi-Sonic Monitor")
        return True