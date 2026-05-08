import sys
import os

# --- FIX PERCORSI ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import gzip
import stable_retro as retro
import pygame
import numpy as np
import config

# Usa la cartella catturati dal config
GAME_PATH = config.CUSTOM_CAPTURE_STATES_DIR
os.makedirs(GAME_PATH, exist_ok=True)


def get_next_index():
    import glob
    files = glob.glob(os.path.join(GAME_PATH, "GHZ_Custom_*.state"))
    if not files: return 1
    indices = []
    for f in files:
        try:
            num = int(os.path.basename(f).split('_')[2].split('.')[0])
            indices.append(num)
        except:
            continue
    return max(indices) + 1 if indices else 1


def run_state_manager():
    env = retro.make(game=config.GAME_NAME, state=config.STATE_NAME, render_mode="rgb_array")
    obs, _ = env.reset()

    pygame.init()
    scale = 3
    screen = pygame.display.set_mode((obs.shape[1] * scale, obs.shape[0] * scale))
    clock = pygame.time.Clock()
    save_idx = get_next_index()

    print(f"--- 🎮 SPARKY STATE MANAGER ---")
    print(f"Salvataggio in: {GAME_PATH}")
    print(f"F5: Salva stato | ESC: Esci")

    while True:
        action = np.zeros(12, dtype=np.int8)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F5:
                    state_name = f"GHZ_Custom_{save_idx}.state"
                    path = os.path.join(GAME_PATH, state_name)
                    with gzip.open(path, 'wb') as f:
                        f.write(env.unwrapped.em.get_state())
                    print(f"✅ STATO SALVATO: {state_name}")
                    save_idx += 1
                if event.key == pygame.K_ESCAPE: return

        keys = pygame.key.get_pressed()
        if keys[pygame.K_d]: action[7] = 1
        if keys[pygame.K_a]: action[6] = 1
        if keys[pygame.K_s]: action[5] = 1
        if keys[pygame.K_SPACE]: action[0] = 1

        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: env.reset()

        image = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(image)
        screen.blit(pygame.transform.scale(surf, (obs.shape[1] * scale, obs.shape[0] * scale)), (0, 0))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    run_state_manager()