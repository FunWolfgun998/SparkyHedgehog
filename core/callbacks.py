import cv2
import numpy as np
import math
from stable_baselines3.common.callbacks import BaseCallback


class ShallyTurboCallback(BaseCallback):
    def __init__(self, render_freq=30, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.show_vision = False
        self.smooth_mode = False  # <-- NUOVA MODALITÀ

        self.ctrl_img = np.zeros((120, 300, 3), dtype=np.uint8)
        cv2.putText(self.ctrl_img, "V = Grid (Veloce)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(self.ctrl_img, "S = Fluido (LENTO!)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _on_step(self):
        cv2.imshow("CONTROLLO", self.ctrl_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):
            self.show_vision, self.smooth_mode = not self.show_vision, False
            if not self.show_vision: cv2.destroyWindow("Multi-Sonic Monitor")
        elif key == ord('s'):
            self.show_vision, self.smooth_mode = True, True
            print("⚠️ MODALITÀ FLUIDA ATTIVA! L'addestramento subirà un rallentamento drastico!")

        if self.show_vision:
            # Se siamo in Smooth, renderizza tutto. Altrimenti ogni tot passi.
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
                        cv2.putText(grid_img, f"#{i}", (c * m_width + 5, r * m_height + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (255, 255, 255), 1)

                    cv2.imshow("Multi-Sonic Monitor", grid_img)
        return True

    def __del__(self):
        cv2.destroyAllWindows()