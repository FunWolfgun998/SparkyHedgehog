import sys
import os

# --- FIX PERCORSI E IMPORTAZIONI ---
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
MODEL_NAME = "Sparky_run_6_90000000.zip"  # Il tuo ultimo modello
MODEL_PATH = os.path.join(config.SAVE_PATH, MODEL_NAME)
TEST_STATE = "GreenHillZone.Act1"

# --- OPZIONI VIDEO ---
SAVE_VIDEO = True
os.makedirs(config.VIDEO_DIR, exist_ok=True)
VIDEO_NAME = os.path.join(config.VIDEO_DIR, f"Test_{MODEL_NAME.replace('.zip', '')}.mp4")


def test():
    print(f"\n--- 🎬 TEST E REGISTRAZIONE VIDEO ---")
    print(f"Modello target: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRORE: File non trovato!")
        return

    # Check Hardware
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"🖥️  Motore di calcolo: {device.upper()}")

    # Setup Ambiente (Uguale al training per non confondere l'IA)
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

    print("\nPremere 'ESC' sulla finestra video per interrompere e salvare.")

    step_counter = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_array, info = env.step(action)
        current_info = info[0]

        # Render e HUD
        raw_frame = env.render()
        frame_with_hud = hud.overlay(raw_frame, current_info)
        frame_bgr = cv2.cvtColor(frame_with_hud, cv2.COLOR_RGB2BGR)

        # Video Recorder
        if SAVE_VIDEO and video_writer is None:
            h, w, _ = frame_bgr.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(VIDEO_NAME, fourcc, 60.0, (w, h))
            print(f"🎥 Registrazione in corso in: {config.VIDEO_DIR}")

        if SAVE_VIDEO:
            video_writer.write(frame_bgr)

        cv2.imshow("Shally Play Test", frame_bgr)

        # 16ms = 60 FPS. Se metti 1ms, il video schizza via a 1000 FPS invisibili!
        key = cv2.waitKey(16) & 0xFF

        if key == 27:
            print("\n🛑 Interrotto dall'utente (Rilevato tasto ESC).")
            break

        # Controlliamo il vero valore vettoriale!
        if done_array[0]:
            print(f"\n🏁 Episodio terminato (Morte o Fine Livello) dopo {step_counter} frames.")
            break

        step_counter += 1

    env.close()
    if video_writer:
        video_writer.release()
        print(f"✅ Video salvato in: {VIDEO_NAME}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()