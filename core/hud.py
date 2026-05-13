import cv2
import numpy as np


class SparkyHUD:
    def __init__(self):
        self.enabled = True
        self.scale = 3
        self.c_text = (255, 255, 255)
        self.c_true = (0, 255, 0)
        self.c_false = (50, 50, 50)
        self.c_warn = (0, 0, 255)

    def toggle(self):
        self.enabled = not self.enabled

    def _draw_bool(self, img, x, y, label, state):
        color = self.c_true if state else self.c_false
        cv2.rectangle(img, (x, y - 12), (x + 12, y), color, -1)
        cv2.putText(img, label, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.c_text, 1)

    def _draw_bar(self, img, x, y, label, value, color):
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.c_text, 1)
        cv2.rectangle(img, (x + 80, y - 10), (x + 180, y), self.c_false, -1)
        cv2.rectangle(img, (x + 80, y - 10), (x + 80 + int(value * 100), y), color, -1)

    def overlay(self, frame, info):
        if not self.enabled:
            return cv2.resize(frame, (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
                              interpolation=cv2.INTER_NEAREST)

        h, w = frame.shape[:2]
        img = cv2.resize(frame, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST)
        H, W = img.shape[:2]

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (220, H), (0, 0, 0), -1)
        cv2.rectangle(overlay, (W - 270, 0), (W, H), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

        # --- PANNELLO SINISTRO ---
        cv2.putText(img, "TELEMETRIA AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y = 60
        stats = [
            f"Speed: {info.get('ground_speed', 0)}", f"Angle: {info.get('angle', 0)}",
            f"Rings: {info.get('rings', 0)}", f"Air: {info.get('air_timer', 1800)}"
        ]
        for s in stats:
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.c_text, 1)
            y += 25

        y += 10
        cv2.putText(img, "SENSORI PERICOLO", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 20
        self._draw_bar(img, 10, y, "Pit:", info.get('ai_pit_danger', 0.0), self.c_warn)
        y += 20
        self._draw_bar(img, 10, y, "Fall:", info.get('ai_falling_danger', 0.0), (0, 165, 255))
        y += 30

        cv2.putText(img, "STATUS BOOLS", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25
        status = int(info.get('status', 0))
        bools = [
            ("Facing Left", status & 1), ("In Air", status & 2),
            ("Rolling", status & 4), ("Invincible", info.get('invincible', 0) > 0),
            ("Pushing Wall", info.get('pushing_wall', 0) > 0)
        ]
        for label, state in bools:
            self._draw_bool(img, 10, y, label, state)
            y += 25

        # --- PANNELLO DESTRO: RADAR VISIVO ---
        rx = W - 260
        cv2.putText(img, "RADAR RAM", (rx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        radar_data = info.get('radar_debug', [])
        ry = 60

        # Ora l'HUD mostra i primi 10 oggetti trovati in memoria
        for i, obj in enumerate(radar_data[:10]):
            if obj.get('id', 0) == 0: continue

            col = self.c_text
            tipo = "OBJ"

            if obj.get('l'):
                col, tipo = self.c_warn, "LETHAL"
            elif obj.get('e'):
                col, tipo = (0, 165, 255), "ENEMY"
            elif obj.get('p'):
                col, tipo = (200, 200, 200), "PLATFORM"
            elif obj.get('it'):
                col, tipo = self.c_true, "ITEM"
            elif obj.get('is_phys'):
                col, tipo = (255, 0, 255), "SPRING"
            elif obj.get('u'):
                col, tipo = (150, 150, 150), "UNKNOWN"  # Nuova categoria per debug!

            # Testo formattato in modo pulito
            cv2.putText(img, f"[{i}] {tipo} (ID:{obj['id']})", (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            ry += 16
            cv2.putText(img, f" Dist: {obj.get('dist', 0)} | dX:{obj['raw_dx']} dY:{obj['raw_dy']}", (rx, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.c_text, 1)
            ry += 25

        return img