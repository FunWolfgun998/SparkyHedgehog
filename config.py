# --- IMPOSTAZIONI DEL GIOCO ---
GAME_NAME = 'SonicTheHedgehog-Genesis-v0'
STATE_NAME = 'GreenHillZone.Act1'

# --- IMPOSTAZIONI DEL SISTEMA ---
NUM_ENVS = 16             # Quanti Sonic far giocare in parallelo

# --- IMPOSTAZIONI DELLA VISIONE ---
IMG_SIZE = 84             # Rimpiccioliamo lo schermo a 84x84 pixel
FRAME_STACK = 4           # Quanti fotogrammi sovrapporre per percepire la velocità
SAVE_PATH = "./models/"
LOG_DIR = "./logs/"