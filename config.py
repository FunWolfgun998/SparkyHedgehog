import os
import glob
import stable_retro as retro

# --- PERCORSI ASSOLUTI (Anti-Errore) ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

GAME_NAME = 'SonicTheHedgehog-Genesis-v0'
STATE_NAME = 'GreenHillZone.Act1'

LOG_DIR = os.path.join(ROOT_DIR, "logs")
CUSTOM_STATES_DIR = os.path.join(ROOT_DIR, "train_states")
CUSTOM_CAPTURE_STATES_DIR = os.path.join(ROOT_DIR, "captured_states")
SAVE_PATH = os.path.join(ROOT_DIR, "models")
VIDEO_DIR = os.path.join(ROOT_DIR, "utils", "video_models")

# --- AUTOMAZIONE RUN ---
run_folders = glob.glob(os.path.join(LOG_DIR, "Round_*"))
RUN_NUMBER = len(run_folders) + 1
CURRENT_RUN_NAME = f"Round_{RUN_NUMBER}"

# --- LOGICA STATI ---
STATE_FILES = glob.glob(os.path.join(CUSTOM_STATES_DIR, "*.state"))
STATES = [os.path.splitext(os.path.basename(f))[0] for f in STATE_FILES]
if len(STATES) == 0:
    STATES = [STATE_NAME]

# --- PARAMETRI HARDWARE ---
NUM_ENVS = 30