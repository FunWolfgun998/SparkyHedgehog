import cv2
import numpy as np


class SparkyHUD:
    def __init__(self):
        self.enabled = True
        self.scale = 3
        self.c_text, self.c_true, self.c_warn = (255, 255, 255), (0, 255, 0), (0, 0, 255)
        self.OBJ_NAMES = {
            14: "Checkpoint", 34: "Buzz Bomber", 37: "Ring", 38: "TV Monitor",
            40: "Motobug", 54: "Spikes", 65: "Spring", 61: "Eggman"  # Aggiungi altri se necessario
        }

    def overlay(self, frame, info):
        if not self.enabled: return cv2.resize(frame, (frame.shape[1] * 3, frame.shape[0] * 3))

        h, w = frame.shape[:2]
        img = cv2.resize(frame, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
        H, W = img.shape[:2]

        # Scurimento pannelli
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (240, H), (0, 0, 0), -1)
        cv2.rectangle(overlay, (W - 270, 0), (W, H), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        # --- PANNELLO SINISTRO (BASE 24) ---
        base_data = info.get('ai_input_vector', [0] * 24)
        cv2.putText(img, "TELEMETRIA IA (24)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        labels = [
            "X Norm", "Y Norm", "VelX", "VelY", "G-Speed", "Angle", "Rings?", "Air", "PitDist",
            "Left", "Air", "Roll", "Invinc", "Shield", "Shoes", "Wall", "BossHP", "Zone", "BossDX", "BossDY",
            "Pit Danger", "Fall Danger", "Water", "ScreenX"
        ]
        for i, val in enumerate(base_data):
            y_pos = 60 + (i * 22)
            color = self.c_true if abs(val) > 0.01 else (100, 100, 100)
            cv2.putText(img, f"{labels[i]}: {val:.2f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # --- PANNELLO DESTRO (RADAR 12 SLOTS) ---
        radar_slots = info.get('ai_radar_slots', [])
        cv2.putText(img, "RADAR IA (12 SLOTS)", (W - 260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for i, obj in enumerate(radar_slots):
            if i >= 12 or obj['id'] == 0: continue
            y_p = 60 + (i * 45)
            name = self.OBJ_NAMES.get(obj['id'], f"ID:{obj['id']}")

            # Colore basato sulla categoria
            color = self.c_warn if obj['l'] else (0, 255, 0) if obj['it'] else self.c_text
            cv2.putText(img, f"[{i}] {name}", (W - 260, y_p), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(img, f" dx:{obj['dx']:.2f} dy:{obj['dy']:.2f} Sc:{obj['score']:.1f}", (W - 260, y_p + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        return img