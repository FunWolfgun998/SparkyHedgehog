import os
import gzip
import stable_retro as retro
import pygame
import numpy as np
import glob
import config  # Importiamo il tuo config

# Usa la cartella definita nel config
GAME_PATH = config.CUSTOM_CAPTURE_STATES_DIR
os.makedirs(GAME_PATH, exist_ok=True)


def get_next_index():
    files = glob.glob(os.path.join(GAME_PATH, "GHZ_Custom_*.state"))
    if not files: return 1
    indices = [int(f.split('_')[-1].split('.')[0]) for f in files if "_" in f]
    return max(indices) + 1 if indices else 1


def run_state_manager():
    # Inizializza con il primo stato disponibile o default
    env = retro.make(game=config.GAME_NAME, state=config.STATE_NAME, render_mode="rgb_array")
    obs, _ = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((obs.shape[1] * 3, obs.shape[0] * 3))
    clock = pygame.time.Clock()
    save_idx = get_next_index()

    print(f"--- SPARKY STATE MANAGER ---")
    print(f"File salvati in: {GAME_PATH}")
    print(f"F5: Salva | ESC: Esci. Prossimo indice: {save_idx}")

    while True:
        action = np.zeros(12, dtype=np.int8)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F5:
                    path = os.path.join(GAME_PATH, f"GHZ_Custom_{save_idx}.state")
                    with gzip.open(path, 'wb') as f:
                        f.write(env.unwrapped.em.get_state())
                    print(f"✅ Salvato in Custom States: GHZ_Custom_{save_idx}.state")
                    save_idx += 1
                if event.key == pygame.K_ESCAPE: return

        keys = pygame.key.get_pressed()
        if keys[pygame.K_d]: action[7] = 1
        if keys[pygame.K_a]: action[6] = 1
        if keys[pygame.K_s]: action[5] = 1
        if keys[pygame.K_SPACE]: action[0] = 1

        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(pygame.transform.scale(surf, (obs.shape[1] * 3, obs.shape[0] * 3)), (0, 0))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    run_state_manager()