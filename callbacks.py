# --- START OF FILE callbacks.py ---

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import config

class ShallyVisionCallback(BaseCallback):
    """
    Crea una griglia modulare (4 colonne per riga) per monitorare Sparky.
    Funziona con qualsiasi numero di NUM_ENVS.
    """
    def __init__(self, render_freq=5, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self):
        if self.num_timesteps % self.render_freq == 0:
            # Estraiamo SOLO la componente visiva del dizionario
            obs = self.locals["new_obs"]["vision"]
            num_envs = obs.shape[0]

            frames =[]
            for i in range(num_envs):
                if len(obs.shape) == 4 and obs.shape[1] == config.FRAME_STACK:
                    f = obs[i, -1, :, :]
                else:
                    f = obs[i, 0, :, -config.IMG_SIZE:]
                frames.append(f)

            cols = 4
            rows = (num_envs + cols - 1) // cols

            while len(frames) < rows * cols:
                frames.append(np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32))

            all_rows =[]
            for r in range(rows):
                start = r * cols
                end = start + cols
                all_rows.append(np.hstack(frames[start:end]))

            grid = np.vstack(all_rows)
            grid_img = (grid * 255).astype(np.uint8)

            display_scale = 2.5
            new_w = int(grid_img.shape[1] * display_scale)
            new_h = int(grid_img.shape[0] * display_scale)
            grid_img = cv2.resize(grid_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            cv2.imshow(f"Mente di Shally V2 ({num_envs} Mondi Multipli)", grid_img)
            cv2.waitKey(1)

        return True