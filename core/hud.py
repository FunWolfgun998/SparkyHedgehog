import cv2
import numpy as np


class SparkyHUD:
    def __init__(self):
        self.enabled = True
        self.scale = 3  # Ingrandisce il frame 320x224 a 960x672 per far leggere i testi
        # Colori BGR per OpenCV
        self.c_text = (255, 255, 255)
        self.c_true = (0, 255, 0)  # Verde fluo
        self.c_false = (50, 50, 50)  # Grigio scuro
        self.c_warn = (0, 0, 255)  # Rosso

    def toggle(self):
        self.enabled = not self.enabled

    def _draw_bool(self, img, x, y, label, state):
        """Disegna un quadratino stile Trackmania per le Booleane"""
        color = self.c_true if state else self.c_false
        cv2.rectangle(img, (x, y - 12), (x + 12, y), color, -1)
        cv2.putText(img, label, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.c_text, 1)

    def overlay(self, frame, info):
        if not self.enabled:
            # Se disabilitato ma dobbiamo comunque fare upscale per la vista grid
            return cv2.resize(frame, (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
                              interpolation=cv2.INTER_NEAREST)

        h, w = frame.shape[:2]
        img = cv2.resize(frame, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST)
        H, W = img.shape[:2]

        # 1. Crea i pannelli laterali semi-trasparenti
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (220, H), (0, 0, 0), -1)  # Pannello Sinistro (Fisica)
        cv2.rectangle(overlay, (W - 260, 0), (W, H), (0, 0, 0), -1)  # Pannello Destro (Radar)
        img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)  # Applica trasparenza

        # --- PANNELLO SINISTRO: FISICA & STATUS ---
        cv2.putText(img, "TELEMETRIA AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        y = 60
        # Dati Continui
        stats = [
            f"X: {info.get('x', 0)}", f"Y: {info.get('y', 0)}",
            f"Vel X: {info.get('velocity_x', 0)}", f"Vel Y: {info.get('velocity_y', 0)}",
            f"Speed: {info.get('ground_speed', 0)}", f"Angle: {info.get('angle', 0)}",
            f"Rings: {info.get('rings', 0)}", f"Air: {info.get('air_timer', 1800)}"
        ]
        for s in stats:
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.c_text, 1)
            y += 25

        y += 10
        cv2.putText(img, "STATUS BOOLS", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 30

        # Dati Booleani
        status = int(info.get('status', 0))
        bools = [
            ("Facing Left", status & 1), ("In Air", status & 2),
            ("Rolling", status & 4), ("Invincible", info.get('invincible', 0) > 0),
            ("Shield", info.get('shield', 0) > 0), ("Shoes", info.get('shoes', 0) > 0),
            ("Pushing Wall", info.get('pushing_wall', 0) > 0),
            ("Underwater",status & 64)
        ]
        for label, state in bools:
            self._draw_bool(img, 10, y, label, state)
            y += 25

        # --- PANNELLO DESTRO: RADAR VISION ---
        rx = W - 250
        cv2.putText(img, "RADAR TARGETS", (rx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        radar_data = info.get('radar_debug', [])
        ry = 60
        for i, obj in enumerate(radar_data[:6]):  # Mostriamo solo i primi 6 oggetti visti dall'IA
            if obj['id'] == 0: continue  # Ignora gli slot vuoti

            # Colora in base alla classificazione dell'IA
            col = self.c_text
            tipo = "Sconosciuto"
            if obj['l']:
                col, tipo = self.c_warn, "LETHAL"
            elif obj['e']:
                col, tipo = (0, 165, 255), "ENEMY"
            elif obj['p']:
                col, tipo = (200, 200, 200), "PLATFORM"
            elif obj['it']:
                col, tipo = self.c_true, "ITEM"
            elif obj['is_phys']:
                col, tipo = (255, 0, 255), "SPRING"

            # Intestazione oggetto
            cv2.putText(img, f"[{i}] {tipo} (ID:{obj['id']})", (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            ry += 18
            # Posizione relativa (DX, DY) come la vede la rete neurale (normalizzata)
            cv2.putText(img, f" dx:{obj['dx']:.2f} dy:{obj['dy']:.2f}", (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        self.c_text, 1)
            ry += 30

        return img