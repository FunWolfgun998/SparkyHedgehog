import os
import cv2
import numpy as np
import math
import time
from stable_baselines3.common.callbacks import BaseCallback
from core.hud import SparkyHUD
import config


class SparkyRoundCheckpoint(BaseCallback):
    def __init__(self, save_freq, save_path, run_number, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.run_number = run_number

    def _on_step(self) -> bool:
        total_steps = self.model.num_timesteps
        offset = config.NUM_ENVS
        if (total_steps // self.save_freq) > ((total_steps - offset) // self.save_freq):
            rounded = (total_steps // self.save_freq) * self.save_freq
            fname = f"Sparky_run_{self.run_number}_{rounded}.zip"
            save_loc = os.path.join(self.save_path, fname)
            self.model.save(save_loc)
            if self.verbose > 0:
                print(f"\n💾 [CHECKPOINT] Modello salvato a {rounded} passi: {fname}")
        return True


class SparkyDirectorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.hud = SparkyHUD()

        # --- STATI DEL DIRETTORE ---
        self.monitor_open = False  # Tasto M: Apre/Chiude il monitor
        self.fluid_mode = False  # Tasto F: 60 FPS reali vs Max Speed
        self.grid_mode = 2  # Tasto G: 1=1x1, 2=2x2, 3=3x3
        self.current_page = 0  # Tasti N/B: Naviga tra gli env
        self.hud_enabled = True  # Tasto H: Toggle HUD
        self.logs_enabled = True  # Tasto P: Print Console On/Off

        self.num_envs = config.NUM_ENVS

        # FIX FINESTRA: WINDOW_AUTOSIZE forza la finestra a rimanere esattamente 400x320
        # Nessun bordo grigio gigante o massimizzazioni accidentali su Linux
        cv2.namedWindow("DIRECTOR CONTROL", cv2.WINDOW_AUTOSIZE)

    def _get_grid_settings(self):
        envs_per_page = self.grid_mode * self.grid_mode
        total_pages = math.ceil(self.num_envs / envs_per_page)
        if self.current_page >= total_pages:
            self.current_page = total_pages - 1
        return envs_per_page, total_pages

    def _draw_control_panel(self):
        # Aumentato leggermente in altezza per far entrare il nuovo comando
        panel = np.zeros((320, 400, 3), dtype=np.uint8)
        envs_per_page, total_pages = self._get_grid_settings()

        c_on = (0, 255, 0)
        c_off = (50, 50, 255)
        c_text = (255, 255, 255)

        cv2.putText(panel, "=== SPARKY DIRECTOR ===", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

        # Status
        m_col = c_on if self.monitor_open else c_off
        cv2.putText(panel, f"M - Monitor: {'APERTO' if self.monitor_open else 'CHIUSO'}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_col, 2)

        f_col = c_on if self.fluid_mode else c_off
        cv2.putText(panel, f"F - Modalita': {'60 FPS (Lento)' if self.fluid_mode else 'MAX SPEED'}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, f_col, 2)

        h_col = c_on if self.hud_enabled else c_off
        cv2.putText(panel, f"H - HUD Telemetria: {'ON' if self.hud_enabled else 'OFF'}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, h_col, 2)

        # NUOVO TASTO P
        p_col = c_on if self.logs_enabled else c_off
        cv2.putText(panel, f"P - Print Console: {'ON' if self.logs_enabled else 'OFF (SILENT)'}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_col, 2)

        cv2.putText(panel, f"G - Griglia: {self.grid_mode}x{self.grid_mode}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    c_text, 2)

        page_info = f"Pagina: {self.current_page + 1}/{total_pages} (Env {self.current_page * envs_per_page}-{(self.current_page + 1) * envs_per_page - 1})"
        cv2.putText(panel, f"N/B - {page_info}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

        cv2.putText(panel, "SPACE - Screenshot Griglia", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return panel

    def _handle_inputs(self, delay):
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('m'):
            self.monitor_open = not self.monitor_open
            if not self.monitor_open:
                try:
                    cv2.destroyWindow("MONITOR")
                except:
                    pass
        if key == ord('f'): self.fluid_mode = not self.fluid_mode
        if key == ord('h'): self.hud_enabled = not self.hud_enabled

        # GESTIONE TASTO P (PRINT)
        if key == ord('p'):
            self.logs_enabled = not self.logs_enabled
            # Invia il comando a tutti i processi paralleli (SparkyReward)
            self.training_env.env_method("set_logger_state", self.logs_enabled)
            print(
                f"\n⚙️ [DIRECTOR] Log su terminale: {'ATTIVI' if self.logs_enabled else 'DISATTIVATI (Performance Mode)'}\n")

        if key == ord('g'):
            self.grid_mode += 1
            if self.grid_mode > 3: self.grid_mode = 1
            self._get_grid_settings()

        envs_per_page, total_pages = self._get_grid_settings()
        if key == ord('n'): self.current_page = (self.current_page + 1) % total_pages
        if key == ord('b'): self.current_page = (self.current_page - 1) % total_pages

        return key

    def _on_step(self):
        delay = 16 if self.fluid_mode and self.monitor_open else 1
        render_freq = 1 if self.fluid_mode else 50

        # Disegna il pannello
        cv2.imshow("DIRECTOR CONTROL", self._draw_control_panel())

        key = self._handle_inputs(delay)

        if self.monitor_open and (self.n_calls % render_freq == 0 or key == 32):
            all_imgs = self.training_env.get_images()
            infos = self.locals.get('infos', [])

            envs_per_page, _ = self._get_grid_settings()
            start_idx = self.current_page * envs_per_page
            end_idx = min(start_idx + envs_per_page, self.num_envs)

            res_w, res_h = 640, 448
            grid_img = np.zeros((self.grid_mode * res_h, self.grid_mode * res_w, 3), dtype=np.uint8)

            self.hud.enabled = self.hud_enabled

            for i, env_idx in enumerate(range(start_idx, end_idx)):
                raw_img = all_imgs[env_idx]
                info = infos[env_idx] if env_idx < len(infos) else {}

                processed = self.hud.overlay(raw_img, info)
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

                # INTER_AREA mantiene il testo nitido quando si rimpicciolisce
                resized = cv2.resize(processed_bgr, (res_w, res_h), interpolation=cv2.INTER_AREA)

                row = i // self.grid_mode
                col = i % self.grid_mode
                grid_img[row * res_h: (row + 1) * res_h, col * res_w: (col + 1) * res_w] = resized

                cv2.putText(grid_img, f"ENV ID: {env_idx}", (col * res_w + 10, row * res_h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("MONITOR", grid_img)

            if key == 32:
                timestamp = time.strftime("%H%M%S")
                filename = os.path.join(config.SCREENSHOT_DIR,
                                        f"scr_{config.CURRENT_RUN_NAME}_stp{self.num_timesteps}_{timestamp}.png")
                cv2.imwrite(filename, grid_img)
                print(f"\n📸 [DIRECTOR] Screenshot salvato: {filename}\n")

        return True