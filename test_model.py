import os
import cv2
import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.wrappers import SparkyDiscretizer, SonicRAMWrapper
import config

os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- CONFIGURAZIONE ---
MODEL_NAME = "sparky_run1_1000000_steps"  # <-- INSERISCI QUI IL NOME DEL TUO SALVATAGGIO
MODEL_PATH = os.path.join(config.SAVE_PATH, f"{MODEL_NAME}.zip")
TEST_STATE = "GreenHillZone.Act1"

# --- OPZIONI VIDEO ---
SAVE_VIDEO = True
VIDEO_NAME = f"Sonic_Test_{MODEL_NAME}.mp4"


def test():
    print(f"--- 🎬 REGISTRAZIONE VIDEO: {MODEL_NAME} ---")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Errore: {MODEL_PATH} non trovato!")
        return

    env = retro.make(game=config.GAME_NAME, state=TEST_STATE, render_mode="rgb_array")
    env = SparkyDiscretizer(env)
    env = SonicRAMWrapper(env)
    env = DummyVecEnv([lambda: env])

    model = PPO.load(MODEL_PATH, env=env, device="cuda")
    obs = env.reset()

    # Preparazione VideoWriter
    video_writer = None
    scale = 3  # Ingrandisce il video (320x224 -> 960x672)

    print("Premi 'ESC' sulla finestra per interrompere il test e salvare il video.")

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        current_info = info[0]

        # Rendering DI OGNI SINGOLO FRAME (Fluidità totale)
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

        # HUD a schermo
        cv2.putText(frame, f"X: {current_info.get('x', 0)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2)
        cv2.putText(frame, f"Eggman HP: {current_info.get('boss_hp', 8)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # Inizializza il file MP4 al primo frame
        if SAVE_VIDEO and video_writer is None:
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec standard
            video_writer = cv2.VideoWriter(VIDEO_NAME, fourcc, 60.0, (w, h))
            print(f"🎥 Registrazione avviata: {VIDEO_NAME}")

        if SAVE_VIDEO:
            video_writer.write(frame)

        cv2.imshow("Shally Play Test", frame)

        # waitKey(1) lo fa andare il più veloce possibile.
        # Se lo vuoi a velocità "umana", metti waitKey(16) (1000ms / 60fps)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        if done:
            print("🏁 Episodio finito!")
            obs = env.reset()
            break  # Esce e salva il video

    env.close()
    if video_writer:
        video_writer.release()
        print(f"✅ Video salvato con successo: {VIDEO_NAME}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()