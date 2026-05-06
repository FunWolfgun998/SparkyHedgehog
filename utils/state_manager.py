import os
import gzip
import stable_retro as retro
import pygame
import numpy as np
import glob

# Carica le info dal tuo config se possibile, o definiscile qui
GAME_NAME = 'SonicTheHedgehog-Genesis-v0'
START_STATE = 'GreenHillZone.Act1'
GAME_PATH = os.path.join(os.path.dirname(retro.__file__), "data", "stable", GAME_NAME)


def get_next_index():
    files = glob.glob(os.path.join(GAME_PATH, "GHZ_Custom_*.state"))
    indices = [int(f.split('_')[-1].split('.')[0]) for f in files if "_" in f]
    return max(indices) + 1 if indices else 1


def run_state_manager():
    env = retro.make(game=GAME_NAME, state=START_STATE, render_mode="rgb_array")
    obs, _ = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((obs.shape[1] * 3, obs.shape[0] * 3))
    clock = pygame.time.Clock()
    save_idx = get_next_index()

    print(f"--- SPARKY STATE MANAGER ---")
    print(f"F5: Salva e Comprime | ESC: Esci. Prossimo indice: {save_idx}")

    while True:
        action = np.zeros(12, dtype=np.int8)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F5:
                    path = os.path.join(GAME_PATH, f"GHZ_Custom_{save_idx}.state")
                    with gzip.open(path, 'wb') as f:
                        f.write(env.unwrapped.em.get_state())
                    print(f"✅ Salvato: GHZ_Custom_{save_idx}.state")
                    save_idx += 1
                if event.key == pygame.K_ESCAPE: return

        keys = pygame.key.get_pressed()
        # Mapping base: W,A,S,D + Space
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