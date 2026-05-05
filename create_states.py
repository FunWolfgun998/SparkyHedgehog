import os
import stable_retro as retro
import pygame
import numpy as np

# --- CONFIGURAZIONE ---
GAME_NAME = 'SonicTheHedgehog-Genesis-v0'
# Puoi cambiare questo per creare stati nell'Act 2 o Act 3
START_STATE = 'GreenHillZone.Act3d a ddddddddddadd'

# Nuova mappatura tasti: WASD + SPAZIO
KEY_MAP = {
    pygame.K_w: 4,  # UP
    pygame.K_a: 6,  # LEFT
    pygame.K_s: 5,  # DOWN
    pygame.K_d: 7,  # RIGHT
    pygame.K_SPACE: 0,  # B (Salto principale)
    pygame.K_RETURN: 3,  # START
}


def main():
    # Inizializza l'emulatore
    # Usiamo rgb_array per catturare i pixel e mostrarli con Pygame
    env = retro.make(game=GAME_NAME, state=START_STATE, render_mode="rgb_array")
    obs, info = env.reset()

    # Inizializza Pygame
    pygame.init()
    # Ingrandiamo la finestra (originale 320x224)
    scale = 3
    screen_size = (obs.shape[1] * scale, obs.shape[0] * scale)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Sparky State Maker - F5 per SALVARE")
    clock = pygame.time.Clock()

    print("\n--- Sparky State Maker ---")
    print("CONTROLLI:")
    print(" - W, A, S, D: Muovi Sonic")
    print(" - SPAZIO: Salto")
    print(" - INVIO: Start")
    print(" - F5: SALVA uno stato (.state)")
    print(" - ESC: Esci\n")

    running = True
    save_count = 46

    while running:
        action = np.zeros(12, dtype=np.int8)

        # Gestione eventi
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_F5:
                    # Percorso salvataggio
                    state_name = f"GHZ_Custom_{save_count}.state"
                    game_path = os.path.join(os.path.dirname(retro.__file__), "data", "stable", GAME_NAME)
                    full_path = os.path.join(game_path, state_name)

                    # Estraiamo lo stato della RAM e scriviamo il file
                    state_data = env.unwrapped.em.get_state()
                    with open(full_path, "wb") as f:
                        f.write(state_data)

                    print(f"✅ STATO {save_count} SALVATO!")
                    print(f"   Nome: {state_name}")
                    print(f"   Percorso: {full_path}\n")
                    save_count += 1

        # Lettura input tastiera continuo
        keys = pygame.key.get_pressed()
        for py_key, retro_btn in KEY_MAP.items():
            if keys[py_key]:
                action[retro_btn] = 1

        # Applica l'azione e ottieni il nuovo frame
        obs, reward, terminated, truncated, info = env.step(action)

        # Se muori o finisce il tempo, lo script resetta per permetterti di continuare a giocare
        if terminated or truncated:
            env.reset()

        # Rendering su Pygame
        # Convertiamo l'array di pixel (H,W,C) in un formato Pygame (W,H,C)
        image = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(image)
        surf = pygame.transform.scale(surf, screen_size)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # 60 FPS per fluidità Genesis originale
        clock.tick(60)

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()