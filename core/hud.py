import cv2
import numpy as np


class SparkyHUD:
    def __init__(self):
        self.enabled = True
        self.scale = 3  # Gioco 320x224 -> HUD 960x672

        self.c_text = (240, 240, 240)
        self.c_true = (50, 255, 50)  # Verde fluo per positivi
        self.c_false = (120, 120, 120)  # Grigio per zeri
        self.c_warn = (50, 50, 255)  # Rosso pericoli
        self.c_title = (0, 215, 255)  # Giallo/Oro per titoli

        self.OBJ_NAMES = {
            # --- HAZARDS & ENEMIES ---
            22: "Harpoon / Missile", 23: "Spiked Helix", 31: "Crabmeat",
            32: "Bomb Enemy", 34: "Buzz Bomber", 35: "Buzz Missile",
            36: "Red Newtron", 40: "Motobug", 43: "Chopper (Fish)",
            44: "Jaws", 45: "Burrobot", 54: "Spikes", 64: "Green Newtron",
            72: "Boss Wrecking Ball", 80: "Yadrin", 87: "Fixed Spikes",
            88: "Moving Spikes", 96: "Orbinaut", 97: "Orbinaut",
            127: "Energy Ball",

            # --- BOSSES ---
            61: "Eggman (GHZ)", 90: "Eggman (SYZ)", 117: "Eggman (Final)",
            118: "Final Machine", 119: "Boss Platform", 121: "Eggman (SLZ)",
            126: "Final Boss",

            # --- PHYSICS ---
            25: "Bumper", 30: "See-Saw", 41: "Spring", 65: "Spring",
            66: "Diag. Spring", 71: "Bumper", 75: "Conveyor Belt",

            # --- PLATFORMS ---
            6: "Ledge Node", 7: "GHZ Rock", 8: "Bridge Stem",
            9: "Floating Log", 11: "Water Bridge", 12: "Wood Bridge",
            15: "Swing Platform", 17: "Tunnel Door", 18: "Moving Block",
            21: "Swing Platform", 24: "Moving Block", 26: "Collapse Ledge",
            28: "Bridge Log", 47: "MZ Press", 49: "MZ Stomper",
            60: "Stone Wall", 62: "Prison Capsule", 68: "Invis. Barrier",
            81: "Smashable Wall", 82: "Collapse Floor",

            # --- ITEMS / VITAL ---
            10: "Air Bubble", 13: "Goal Signpost", 14: "Checkpoint",
            37: "Ring", 38: "TV Monitor"
        }

    def toggle(self):
        self.enabled = not self.enabled

    def overlay(self, frame, info):
        if not self.enabled:
            return cv2.resize(frame, (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
                              interpolation=cv2.INTER_NEAREST)

        h, w = frame.shape[:2]
        img = cv2.resize(frame, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST)
        H, W = img.shape[:2]

        # Pannelli semi-trasparenti SCURITI (92% neri) per contrasto estremo
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (220, H), (0, 0, 0), -1)  # Sinistra compattata
        cv2.rectangle(overlay, (W - 290, 0), (W, H), (0, 0, 0), -1)  # Destra radar
        img = cv2.addWeighted(overlay, 0.92, img, 0.08, 0)

        # ==========================================
        # PANNELLO SINISTRO: I 24 SENSORI
        # ==========================================
        base_data = info.get('ai_input_vector', [0.0] * 24)
        cv2.putText(img, "IA BASE SENSORS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.c_title, 2)

        # Nomi puliti senza numeri davanti
        labels = [
            "X Norm", "Y Norm", "Vel X", "Vel Y", "G-Speed", "Angle",
            "Rings(B)", "Air Timer", "Pit Dist", "Left(B)", "Air(B)",
            "Roll(B)", "Invinc(B)", "Shield(B)", "Shoes(B)", "Wall(B)",
            "Boss HP", "Zone", "Boss DX", "Boss DY", "Danger Pit",
            "Danger Fall", "Water(B)", "Screen X"
        ]

        y_start = 60
        y_step = 24

        for i in range(min(24, len(base_data))):
            val = base_data[i]
            y_pos = y_start + (i * y_step)

            if val == 0.0:
                color, txt_val = self.c_false, " 0.00"
            elif val == 1.0:
                color, txt_val = self.c_true, "+1.00"
            else:
                color = self.c_text if val > 0 else (180, 180, 255)
                txt_val = f"{val:+.2f}"

            # Disegna Nome e Valore molto più vicini e allineati (X: 10 vs X: 140)
            cv2.putText(img, labels[i], (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            cv2.putText(img, txt_val, (140, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        # ==========================================
        # PANNELLO DESTRO: RADAR A 12 SLOT
        # ==========================================
        radar_slots = info.get('ai_radar_slots', [])
        cv2.putText(img, "IA RADAR SLOTS", (W - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.c_title, 2)

        ry_start = 60
        ry_step = 48  # Spazio vitale per slot (2 righe di testo per slot)

        for i in range(12):
            y_p = ry_start + (i * ry_step)

            if i < len(radar_slots) and radar_slots[i]['id'] != 0:
                obj = radar_slots[i]
                name = self.OBJ_NAMES.get(obj['id'], f"OBJ_ID:{obj['id']}")

                # Colore e tag prioritario
                if obj['l']:
                    cat_color, tag = self.c_warn, "[LETHAL]"
                elif obj['e']:
                    cat_color, tag = (0, 165, 255), "[ENEMY]"
                elif obj['ph']:
                    cat_color, tag = (255, 0, 255), "[PHYSIC]"
                elif obj['it']:
                    cat_color, tag = (0, 255, 255), "[ITEM]"
                elif obj['p']:
                    cat_color, tag = (180, 180, 180), "[PLAT]"
                else:
                    cat_color, tag = self.c_text, "[VITAL]"

                # Riga 1: Index, Nome allineato a sinistra, Tag allineato a destra
                cv2.putText(img, f"#{i + 1} {name}", (W - 280, y_p), cv2.FONT_HERSHEY_SIMPLEX, 0.48, cat_color, 1)
                cv2.putText(img, tag, (W - 90, y_p), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cat_color, 1)

                # Riga 2: Distanze e Flag vettoriali
                flags = f"L:{obj['l']} E:{obj['e']} P:{obj['p']} I:{obj['it']}"
                cv2.putText(img, f"dx:{obj['dx']:+.2f} dy:{obj['dy']:+.2f} | {flags}", (W - 280, y_p + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            else:
                # Slot vuoto (Zeri)
                cv2.putText(img, f"#{i + 1} [ EMPTY SLOT ]", (W - 280, y_p), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            self.c_false, 1)
                cv2.putText(img, "dx:+0.00 dy:+0.00 | L:0 E:0 P:0 I:0", (W - 280, y_p + 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, self.c_false, 1)

        return img