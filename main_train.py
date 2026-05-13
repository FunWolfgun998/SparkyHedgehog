import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import torch
from stable_baselines3 import PPO
from core.environment import create_parallel_envs
from core.callbacks import ShallyTurboCallback, SparkyRoundCheckpoint
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

    # Architettura della rete (3 strati da 512 neuroni)
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512])
    )

    RESUME_MODEL = ""

    if os.path.exists(RESUME_MODEL) and RESUME_MODEL != "":
        print(f"🔄 CARICAMENTO MODELLO: {RESUME_MODEL}")
        model = PPO.load(RESUME_MODEL, env=envs, device="cuda")
    else:
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
            learning_rate=linear_schedule(2e-4),  # Scheduler attivo
            gamma=0.998,
            ent_coef=0.05,  # Esplorazione alta all'inizio
            tensorboard_log=config.LOG_DIR
        )

    checkpoint = SparkyRoundCheckpoint(save_freq=1_000_000, save_path=config.SAVE_PATH, run_number=config.RUN_NUMBER)
    vision = ShallyTurboCallback(render_freq=50)

    model.learn(
        total_timesteps=80_000_000,
        callback=[checkpoint, vision],
        tb_log_name=config.CURRENT_RUN_NAME,
        reset_num_timesteps=False
    )


if __name__ == "__main__":
    main()