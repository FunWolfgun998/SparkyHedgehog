import cv2
import numpy as np
import math
from stable_baselines3.common.callbacks import BaseCallback


class ShallyTurboCallback(BaseCallback):
    def __init__(self, render_freq=10, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.show_vision = False

        # Finestra di controllo
        self.ctrl_img = np.zeros((120, 300, 3), dtype=np.uint8)
        cv2.putText(self.ctrl_img, "CLICCA QUI E", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(self.ctrl_img, "PREMI 'V' PER GRID", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
            print(f"\n[GRID VISION] {'ATTIVATA' if self.show_vision else 'DISATTIVATA'}")

        if self.show_vision:
            if self.n_calls % self.render_freq == 0:
                # Prendi le immagini di TUTTI gli ambienti
                imgs = self.training_env.get_images()
                if imgs is not None and len(imgs) > 0:
                    num_envs = len(imgs)

                    # Calcoliamo quante righe e colonne servono (es. 24 env -> 6 col x 4 righe)
                    cols = math.ceil(math.sqrt(num_envs))
                    rows = math.ceil(num_envs / cols)

                    # Dimensione di ogni singola miniatura (piccole per non saturare la GPU)
                    m_width, m_height = 320, 224

                    # Creiamo il canvas nero per la griglia
                    grid_img = np.zeros((rows * m_height, cols * m_width, 3), dtype=np.uint8)

                    for i, img in enumerate(imgs):
                        # Converti e rimpicciolisci ogni Sonic
                        small_frame = cv2.resize(img, (m_width, m_height), interpolation=cv2.INTER_NEAREST)
                        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR)

                        # Calcola posizione nella griglia
                        r = i // cols
                        c = i % cols

                        # Inserisci il piccolo Sonic nel mosaico
                        grid_img[r * m_height:(r + 1) * m_height, c * m_width:(c + 1) * m_width] = small_frame
                        # Opzionale: scrivi il numero dell'env sopra ogni miniatura
                        cv2.putText(grid_img, f"#{i}", (c * m_width + 5, r * m_height + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.imshow("Multi-Sonic Monitor", grid_img)

        return True

    def __del__(self):
        cv2.destroyAllWindows()