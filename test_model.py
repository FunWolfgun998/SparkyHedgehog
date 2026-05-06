import os
import cv2
import torch
import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO
from core.wrappers import SparkyDiscretizer, SonicRAMWrapper
import config
os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- CONFIGURAZIONE TEST ---
# Inserisci qui il nome del file che vuoi testare (quello dentro la cartella models)
MODEL_NAME = "sparky_ram_4499280_steps"
MODEL_PATH = os.path.join(config.SAVE_PATH, f"{MODEL_NAME}.zip")

# Livello da testare
TEST_STATE = "GreenHillZone.Act1" 

def test():
    print(f"--- 🎮 TEST MODELLO: {MODEL_NAME} ---")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Errore: Il file {MODEL_PATH} non esiste!")
        return

    # Creazione ambiente singolo
    # Usiamo rgb_array per poter ingrandire la finestra con OpenCV
    env = retro.make(game=config.GAME_NAME, state=TEST_STATE, render_mode="rgb_array")
    
    # Applichiamo gli STESSI wrapper usati in training
    # Nota: SparkyReward non serve per il test (non dobbiamo allenare)
    env = SparkyDiscretizer(env)
    env = SonicRAMWrapper(env)

    # Caricamento Modello
    model = PPO.load(MODEL_PATH, env=env, device="cuda")

    obs, info = env.reset()
    
    print("Premi 'ESC' sulla finestra del video per chiudere il test.")
    
    done = False
    while not done:
        # deterministic=True è FONDAMENTALE per il test
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Rendering con OpenCV (per vedere Sonic grande)
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Ingrandiamo il video di 3 volte
        scale = 3
        frame = cv2.resize(frame, (frame.shape[1]*scale, frame.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        
        # Mostriamo alcune info a schermo
        cv2.putText(frame, f"X: {info.get('x', 0)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"HP Boss: {info.get('boss_hp', 0)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Shally Play Test", frame)

        # Gestione chiusura
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break
            
        if terminated or truncated:
            print("Episodio finito. Reset...")
            obs, info = env.reset()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()