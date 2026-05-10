import sys
import os

# --- FIX PERCORSI E IMPORTAZIONI ---
# Aggiungiamo la root directory al path di Python così trova 'core' e 'config'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import torch
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.wrappers import SparkyDiscretizer, SonicRAMWrapper
import config

# --- CONFIGURAZIONE ---
MODEL_NAME = "Sparky_run_2_110000000.zip"  # <-- INSERISCI QUI IL NOME DEL SALVATAGGIO CON IL .ZIP
MODEL_PATH = os.path.join(config.SAVE_PATH, MODEL_NAME)
TEST_STATE = "GreenHillZone.Act1"

# --- OPZIONI VIDEO ---
SAVE_VIDEO = True
os.makedirs(config.VIDEO_DIR, exist_ok=True) # Crea la cartella se non esiste
VIDEO_NAME = os.path.join(config.VIDEO_DIR, f"Test_{MODEL_NAME.replace('.zip', '')}.mp4")

def test():
    print(f"\n--- 🎬 TEST E REGISTRAZIONE VIDEO ---")
    print(f"Modello target: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRORE: File non trovato!")
        return

    # --- CHECK HARDWARE UNIVERSALE ---
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    print(f"🖥️  Motore di calcolo: {device.upper()}")

    # Setup Ambiente
    env = retro.make(game=config.GAME_NAME, state=TEST_STATE, render_mode="rgb_array")
    env = SparkyDiscretizer(env)
    env = SonicRAMWrapper(env)
    env = DummyVecEnv([lambda: env])

    try:
        model = PPO.load(MODEL_PATH, env=env, device=device)
        print("✅ Modello caricato con successo!")
    except Exception as e:
        print(f"❌ Errore caricamento: {e}")
        return

    obs = env.reset()
    video_writer = None
    scale = 3

    print("\nPremere 'ESC' sulla finestra video per interrompere e salvare.")

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        current_info = info[0]

        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

        # HUD
        cv2.putText(frame, f"X: {current_info.get('x', 0)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Eggman HP: {current_info.get('boss_hp', 8)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Video Recorder
        if SAVE_VIDEO and video_writer is None:
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(VIDEO_NAME, fourcc, 60.0, (w, h))
            print(f"🎥 Registrazione in corso in: {config.VIDEO_DIR}")

        if SAVE_VIDEO:
            video_writer.write(frame)

        cv2.imshow("Shally Play Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        if done:
            print("🏁 Episodio completato.")
            break

    env.close()
    if video_writer:
        video_writer.release()
        print(f"✅ Video salvato in: {VIDEO_NAME}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()