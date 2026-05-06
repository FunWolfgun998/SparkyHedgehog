import os

# --- OTTIMIZZAZIONI CACHYOS / AMD ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
# ------------------------------------

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from core.environment import create_parallel_envs
from core.callbacks import ShallyTurboCallback
import config


def main():
    print("--- 🦔 INIZIALIZZAZIONE SPARKYHEDGEHOG (RAM BASED) ---")
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.SAVE_PATH, exist_ok=True)

    # Verifica GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Dispositivo utilizzato: {device}")

    envs = create_parallel_envs()

    # --- ARCHITETTURA DEL CERVELLO (Il cuore del successo) ---
    # Usiamo 3 strati da 256 neuroni. È il bilanciamento perfetto tra
    # potenza di calcolo (GPU) e capacità logica per finire i livelli.
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512])
    )

    RESUME_MODEL = "models/sparky_ram_3999360_steps.zip"  # Inserisci il percorso .zip per riprendere

    if RESUME_MODEL and os.path.exists(RESUME_MODEL):
        print(f"🔄 SBLOCCO SPARKY DA: {RESUME_MODEL}")
        model = PPO.load(
            RESUME_MODEL,
            env=envs,
            custom_objects={
                "learning_rate": 5e-5,  # Più lento per non distruggere ciò che sa
                "ent_coef": 0.05,  # RADDOPPIA LA CURIOSITÀ (da 0.05 a 0.1)
                "n_steps": 4096  # Accumula più prove prima di decidere
            }
        )
    else:
        print("🧠 CREAZIONE NUOVO CERVELLO OTTIMIZZATO...")
        model = PPO(
            "MlpPolicy",
            envs,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=config.LOG_DIR,
            device=device,

            # Parametri bilanciati per finire i livelli in 2 settimane
            learning_rate=0.00025,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.995,  # Fondamentale per Sonic (guarda al futuro)
            gae_lambda=0.95,
            ent_coef=0.03,  # Esplorazione moderata
            clip_range=0.2,
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000_000 // config.NUM_ENVS, 1),
        save_path=config.SAVE_PATH,
        name_prefix="sparky_ram"
    )
    # Mostra un frame ogni 20 passi se la visione è attiva
    vision_callback = ShallyTurboCallback(render_freq=20)

    print("🚀 PROGETTO AVVIATO. Clicca sulla finestrella 'CONTROLLO' e premi 'V' per il video.")

    model.learn(
        total_timesteps=50_000_000,
        callback=[checkpoint_callback, vision_callback],
        tb_log_name="SparkyRun_RAM",
        reset_num_timesteps=False
    )

    model.save(os.path.join(config.SAVE_PATH, "sparky_final_ram"))
    envs.close()


if __name__ == "__main__":
    main()