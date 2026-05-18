import os

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import torch
from stable_baselines3 import PPO
from core.environment import create_parallel_envs
from core.callbacks import SparkyDirectorCallback, SparkyRoundCheckpoint  # <-- MODIFICATO IMPORT
import config


# --- SCHEDULER PER IL LEARNING RATE ---
def linear_schedule(initial_value, final_value=0.0):
    """Decresce linearmente un valore da initial a final."""
    def func(progress_remaining):
        # progress_remaining va da 1.0 (inizio) a 0.0 (fine)
        return final_value + (initial_value - final_value) * progress_remaining
    return func


def main():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.SAVE_PATH, exist_ok=True)

    print(f"--- 🦔 SPARKYHEDGEHOG: SISTEMA RIVISTO ---")

    envs = create_parallel_envs()

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(
            pi=[512, 512],  # Attore più agile
            vf=[512, 512]  # Critico più potente
        )
    )

    MODEL_NAME = "Sparky_run_8_228000000.zip"
    RESUME_MODEL = os.path.join(config.SAVE_PATH, MODEL_NAME)

    # Parametri di addestramento dinamici
    lr_schedule = linear_schedule(3e-5, 1e-6)  # Il passo si fa più piccolo e preciso

    if os.path.isfile(RESUME_MODEL):
        print(f"♻️ Ripristino modello esistente: {MODEL_NAME}")
        model = PPO.load(RESUME_MODEL, env=envs, device="cuda",
                         learning_rate=lr_schedule, ent_coef=0.01, gamma = 0.998,
                         tensorboard_log=config.LOG_DIR)
    else:
        print("🆕 Nessun modello trovato. Creazione di una nuova rete neurale...")
        model = PPO(
            "MlpPolicy",
            envs,
            policy_kwargs=policy_kwargs,
            device="cuda",
            verbose=1,
            n_steps=4096,
            batch_size=1024,
            n_epochs=15,
            learning_rate=lr_schedule,
            ent_coef=0.02,  # Esplorazione iniziale
            gamma=0.998,
            tensorboard_log=config.LOG_DIR
        )

    checkpoint = SparkyRoundCheckpoint(save_freq=1_000_000, save_path=config.SAVE_PATH, run_number=config.RUN_NUMBER)

    # Inizializziamo il Director Mode
    director = SparkyDirectorCallback()

    model.learn(
        total_timesteps=80_000_000,
        callback=[checkpoint, director],
        tb_log_name=config.CURRENT_RUN_NAME,
        reset_num_timesteps=False
    )


if __name__ == "__main__":
    main()