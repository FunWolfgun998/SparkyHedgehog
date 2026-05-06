import os
import glob
import stable_retro as retro

# --- IMPOSTAZIONI DEL GIOCO ---
GAME_NAME = 'SonicTheHedgehog-Genesis-v0'
STATE_NAME = 'GreenHillZone.Act1' # <-- FONDAMENTALE: Lo stato iniziale di default

GAME_DATA_PATH = os.path.join(os.path.dirname(retro.__file__), "data", "stable", GAME_NAME)
STATE_FILES = glob.glob(os.path.join(GAME_DATA_PATH, "*.state"))
STATES =[os.path.splitext(os.path.basename(f))[0] for f in STATE_FILES]

# --- IMPOSTAZIONI DEL SISTEMA ---
NUM_ENVS = 24 # Abbassa a 16 se la CPU sta fissa al 100% e lagga il PC
SAVE_PATH = "./models/"
LOG_DIR = "./logs/"