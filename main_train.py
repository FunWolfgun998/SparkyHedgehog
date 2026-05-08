import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from core.environment import create_parallel_envs
from core.callbacks import ShallyTurboCallback
import config

class SparkyRoundCheckpoint(BaseCallback):
    def __init__(self, save_freq, save_path, run_number):
        super().__init__()
        self.save_freq, self.save_path, self.run_number = save_freq, save_path, run_number

    def _on_step(self) -> bool:
        total_steps = self.model.num_timesteps
        # Salva a ogni multiplo esatto del milione
        if (total_steps // self.save_freq) > ((total_steps - config.NUM_ENVS) // self.save_freq):
            rounded = (total_steps // self.save_freq) * self.save_freq
            fname = f"Sparky_run_{self.run_number}_{rounded}.zip"
            self.model.save(os.path.join(self.save_path, fname))
            print(f"\n💾 BACKUP TONDO: {fname}")
        return True

def main():
    # 1. Creazione cartelle sicura (viene eseguita solo dal processo padre)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CUSTOM_STATES_DIR, exist_ok=True)
    os.makedirs(config.SAVE_PATH, exist_ok=True)

    # 2. Stampe singole di info
    print(f"--- 🦔 INIZIALIZZAZIONE SPARKYHEDGEHOG ---")
    if len(config.STATES) == 1 and config.STATES[0] == config.STATE_NAME:
        print(f"⚠️ Cartella train_states/ vuota o inesistente. Uso default: {config.STATE_NAME}")
    else:
        print(f"✅ RUN {config.RUN_NUMBER} | Caricati {len(config.STATES)} stati per l'addestramento.")

    # 3. Check Hardware Universale
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" # Per Mac Apple Silicon (se mai cambierai PC)
    else:
        device = "cpu"
    print(f"🖥️  Dispositivo utilizzato: {device.upper()}")

    envs = create_parallel_envs()
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))

    RESUME_MODEL = "models/Sparky_run_9_182000000.zip"

    if os.path.exists(RESUME_MODEL):
        print(f"🔄 SBLOCCO RUN {config.RUN_NUMBER} DA {RESUME_MODEL}")
        model = PPO.load(RESUME_MODEL, env=envs, device="cuda",
                         custom_objects={"learning_rate": 2e-5, "ent_coef": 0.05, "n_steps": 8192, "batch_size": 2048})
    else:
        model = PPO("MlpPolicy", envs, policy_kwargs=policy_kwargs, device="cuda", verbose=1,
                    n_steps=4096, batch_size=1024, n_epochs=15, learning_rate=2e-4,
                    gamma=0.998, ent_coef=0.03, tensorboard_log=config.LOG_DIR)

    checkpoint = SparkyRoundCheckpoint(save_freq=1000000, save_path=config.SAVE_PATH, run_number=config.RUN_NUMBER)
    vision = ShallyTurboCallback(render_freq=50)

    model.learn(total_timesteps=80_000_000, callback=[checkpoint, vision],
                tb_log_name=config.CURRENT_RUN_NAME, reset_num_timesteps=False)

if __name__ == "__main__":
    main()