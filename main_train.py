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
def linear_schedule(initial_value):
    """Decresce linearmente il tasso di apprendimento."""

    def func(progress_remaining):
        return progress_remaining * initial_value

    return func


def main():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.SAVE_PATH, exist_ok=True)

    print(f"--- 🦔 SPARKYHEDGEHOG: SISTEMA RIVISTO ---")

    envs = create_parallel_envs()

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512])
    )

    MODEL_NAME = "Sparky_run_1_2000000.zip"
    RESUME_MODEL = os.path.join(config.SAVE_PATH, MODEL_NAME)

    if os.path.exists(RESUME_MODEL):
        print(f"🔄 CARICAMENTO MODELLO: {RESUME_MODEL}")
        model = PPO.load(RESUME_MODEL, env=envs, device="cuda")
    else:
        print(f"⚠️ MODELLO NON TROVATO a: {RESUME_MODEL}")
        print("🆕 CREAZIONE NUOVA RETE NEURALE (188 Inputs)")
        model = PPO(
            "MlpPolicy",
            envs,
            policy_kwargs=policy_kwargs,
            device="cuda",
            verbose=1,
            n_steps=4096,
            batch_size=1024,
            n_epochs=15,
            learning_rate=linear_schedule(2e-4),
            gamma=0.998,
            ent_coef=0.05,
            tensorboard_log=config.LOG_DIR
        )

    checkpoint = SparkyRoundCheckpoint(save_freq=1_000_000, save_path=config.SAVE_PATH, run_number=config.RUN_NUMBER)

    # Inizializziamo il Director Mode
    director = SparkyDirectorCallback()

    model.learn(
        total_timesteps=8_000_000,
        callback=[checkpoint, director],
        tb_log_name=config.CURRENT_RUN_NAME,
        reset_num_timesteps=False
    )


if __name__ == "__main__":
    main()