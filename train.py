# train.py
import os

# --- CONFIGURAZIONE GPU AMD (Deve stare in cima!) ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["SDL_VIDEODRIVER"] = "dummy"
# ----------------------------------------------------

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from environment import create_parallel_envs
from network import Shally
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
            obs = self.locals["new_obs"]
            num_envs = obs.shape[0]

            frames =[]
            for i in range(num_envs):
                # SB3 impila sui canali. La forma è (N, 4, 84, 84).
                # Prendiamo l'ultimo canale (-1) che è il frame più recente.
                if len(obs.shape) == 4 and obs.shape[1] == config.FRAME_STACK:
                    f = obs[i, -1, :, :]
                else:
                    # Fallback di sicurezza
                    f = obs[i, 0, :, -config.IMG_SIZE:]

                frames.append(f)

            # Parametri della griglia
            cols = 4
            rows = (num_envs + cols - 1) // cols

            # Riempiamo i posti vuoti con immagini nere
            while len(frames) < rows * cols:
                frames.append(np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32))

            # Creiamo la griglia
            all_rows =[]
            for r in range(rows):
                start = r * cols
                end = start + cols
                row_chunk = np.hstack(frames[start:end])
                all_rows.append(row_chunk)

            grid = np.vstack(all_rows)
            grid_img = (grid * 255).astype(np.uint8)

            # Ingrandiamo la finestra
            display_scale = 2.5
            new_w = int(grid_img.shape[1] * display_scale)
            new_h = int(grid_img.shape[0] * display_scale)

            grid_img = cv2.resize(grid_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            cv2.imshow(f"Mente di Shally ({num_envs} Mondi)", grid_img)
            cv2.waitKey(1)

        return True


def main():
    print("--- INIZIALIZZAZIONE PROGETTO SPARKYHEDGEHOG ---")
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.SAVE_PATH, exist_ok=True)

    envs = create_parallel_envs()

    policy_kwargs = dict(
        features_extractor_class=Shally,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # ============================================================
    # IMPOSTA QUI IL NOME DEL FILE PER RIPARTIRE DA UN SALVATAGGIO
    # Se lo lasci vuoto (""), inizierà un addestramento da zero.
    # ============================================================
    RESUME_PATH = "models/shally_final_model.zip"  # Esempio: "models/shally_checkpoint_199992_steps"

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        print(f"Bentornata Shally! Caricamento ricordi da: {RESUME_PATH}")
        model = PPO.load(RESUME_PATH, env=envs, tensorboard_log=config.LOG_DIR)
    else:
        print("Innestando una nuova Shally nel modello PPO...")
        model = PPO(
            policy="CnnPolicy",
            env=envs,
            device="cuda",
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=config.LOG_DIR,
            learning_rate=0.0001,
            n_steps=4096,  # Molti dati per la GPU AMD
            batch_size=1024,  # Sfrutta i core della RX 7600 XT
            n_epochs=10,
            gamma=0.999,  # Visione a lunghissimo termine
            ent_coef=0.08,  # ALTA CURIOSITÀ per superare la rampa
            clip_range=0.2,
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // config.NUM_ENVS, 1),
        save_path=config.SAVE_PATH,
        name_prefix="shally_checkpoint"
    )
    vision_callback = ShallyVisionCallback(render_freq=5)

    total_steps = 2_000_000

    print(f"--- AVVIO ADDESTRAMENTO ({total_steps} passi) ---")
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_callback, vision_callback],
        reset_num_timesteps=False,
        tb_log_name = "SparkyRun"
    )

    final_model_path = os.path.join(config.SAVE_PATH, "shally_final_model")
    model.save(final_model_path)
    print(f"Addestramento completato! Cervello salvato in: {final_model_path}")
    envs.close()


if __name__ == "__main__":
    main()