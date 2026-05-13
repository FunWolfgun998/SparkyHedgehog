import cv2
import numpy as np


class SparkyHUD:
    def __init__(self):
        self.enabled = True
        self.scale = 3  # Gioco 320x224 -> HUD 960x672

        # Palette Colori ad Alto Contrasto
        self.c_text = (240, 240, 240)
        self.c_true = (50, 255, 50)
        self.c_false = (100, 100, 100)
        self.c_warn = (50, 50, 255)
        self.c_title = (0, 215, 255)

        self.OBJ_NAMES = {
            14: "Lamppost", 34: "Buzz Bomber", 37: "Ring", 38: "Monitor",
            40: "Motobug", 54: "Spikes", 65: "Spring", 61: "Eggman"
        }

    def toggle(self):
        self.enabled = not self.enabled

    def overlay(self, frame, info):
        # Se spento, fa solo l'upscale per non rompere il layout della griglia
        if not self.enabled: return cv2.resize(frame, (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
                                               interpolation=cv2.INTER_NEAREST)

        h, w = frame.shape[:2]
        img = cv2.resize(frame, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST)
        H, W = img.shape[:2]

        # Pannelli semi-trasparenti: Più larghi per far entrare bene i testi leggibili
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (260, H), (0, 0, 0), -1)  # Sinistra (24 Inputs)
        cv2.rectangle(overlay, (W - 320, 0), (W, H), (0, 0, 0), -1)  # Destra (12 Radar)
        img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)  # Molto scuro per leggibilità massima

        # --- PANNELLO SINISTRO: IL VETTORE BASE (INPUT 1 - 24) ---
        base_data = info.get('ai_input_vector', [0.0] * 24)
        cv2.putText(img, "IA VEC_1D (BASE 24)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.c_title, 2)

        labels = [
            "01 X Norm", "02 Y Norm", "03 Vel X", "04 Vel Y", "05 G-Speed", "06 Angle",
            "07 Rings(Bool)", "08 Air Timer", "09 Pit Dist", "10 Left(Bool)", "11 Air(Bool)",
            "12 Roll(Bool)", "13 Invinc(Bool)", "14 Shield(Bool)", "15 Shoes(Bool)", "16 Wall(Bool)",
            "17 Boss HP", "18 Zone", "19 Boss DX", "20 Boss DY", "21 Danger Pit",
            "22 Danger Fall", "23 Water(Bool)", "24 Screen X"
        ]

        # Spaziatura dinamica per far entrare 24 righe in 672 pixel
        y_start = 60
        y_step = 24

        for i in range(min(24, len(base_data))):
            val = base_data[i]
            y_pos = y_start + (i * y_step)

            # Formattazione condizionale: Se è Bool o vicino a zero
            if val == 0.0:
                color = self.c_false
                txt_val = "0.00"
            elif val == 1.0:
                color = self.c_true
                txt_val = "1.00"
            else:
                color = self.c_text if val > 0 else (200, 200, 200)
                txt_val = f"{val:+.2f}"

            # Disegna label e valore allineati
            cv2.putText(img, labels[i], (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(img, txt_val, (190, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # --- PANNELLO DESTRO: IL RADAR (INPUT 25 - 132) ---
        radar_slots = info.get('ai_radar_slots', [])
        cv2.putText(img, "RADAR IA (12 SLOTS x 9)", (W - 310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.c_title, 2)

        ry_start = 60
        ry_step = 48  # Spazio vitale per slot (2 righe di testo per slot)

        for i in range(12):
            y_p = ry_start + (i * ry_step)

            if i < len(radar_slots) and radar_slots[i]['id'] != 0:
                obj = radar_slots[i]
                name = self.OBJ_NAMES.get(obj['id'], f"OBJ_ID:{obj['id']}")

                # Colore per classe per lettura immediata
                if obj['l']:
                    cat_color, tag = self.c_warn, "[LETHAL]"
                elif obj['e']:
                    cat_color, tag = (0, 165, 255), "[ENEMY]"
                elif obj['ph']:
                    cat_color, tag = (255, 0, 255), "[PHYSIC]"
                elif obj['it']:
                    cat_color, tag = (0, 255, 255), "[ITEM]"
                elif obj['p']:
                    cat_color, tag = (180, 180, 180), "[PLATFORM]"
                else:
                    cat_color, tag = self.c_text, "[VITAL]"

                # Riga 1: Index, Nome, Tag e Priorità
                cv2.putText(img, f"#{i + 1} {name} {tag}", (W - 310, y_p), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cat_color, 1)

                # Riga 2: Esattamente i tensori neurali: dx, dy e Flags attivi
                flags = f"L:{obj['l']} E:{obj['e']} P:{obj['p']} I:{obj['it']}"
                cv2.putText(img, f"dx:{obj['dx']:+.2f} dy:{obj['dy']:+.2f} | {flags}", (W - 310, y_p + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                # Slot vuoto (Padding) - Esattamente quello che vede la rete neurale (zeri)
                cv2.putText(img, f"#{i + 1} [ SLOT VUOTO ]", (W - 310, y_p), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            self.c_false, 1)
                cv2.putText(img, "dx:+0.00 dy:+0.00 | L:0 E:0 P:0 I:0", (W - 310, y_p + 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, self.c_false, 1)

        return img