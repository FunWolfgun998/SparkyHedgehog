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
from core.hud import SparkyHUD
import config

# --- CONFIGURAZIONE ---
MODEL_NAME = "Sparky_run_1_4000000.zip"  # <-- INSERISCI QUI IL NOME DEL SALVATAGGIO CON IL .ZIP
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
    hud = SparkyHUD()
    scale = 3

    print("\nPremere 'ESC' sulla finestra video per interrompere e salvare.")

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        current_info = info[0]

        # Estrai l'immagine grezza
        raw_frame = env.render()

        # Applica l'HUD professionale!
        frame_with_hud = hud.overlay(raw_frame, current_info)

        # Converti per OpenCV
        frame_bgr = cv2.cvtColor(frame_with_hud, cv2.COLOR_RGB2BGR)

        # Video Recorder (Ora registra il frame con l'HUD ad alta risoluzione)
        if SAVE_VIDEO and video_writer is None:
            h, w, _ = frame_bgr.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(VIDEO_NAME, fourcc, 60.0, (w, h))
            print(f"🎥 Registrazione in corso in: {config.VIDEO_DIR}")

        if SAVE_VIDEO:
            video_writer.write(frame_bgr)

        cv2.imshow("Shally Play Test", frame_bgr)

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
