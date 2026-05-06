import os
import glob
import stable_retro as retro

# --- SETTINGS GIOCO ---
GAME_NAME = 'SonicTheHedgehog-Genesis-v0'
STATE_NAME = 'GreenHillZone.Act1'

# --- AUTOMAZIONE RUN ---
LOG_DIR = "./logs/"
os.makedirs(LOG_DIR, exist_ok=True)
# Conta le cartelle Round_ per decidere il numero della run
run_folders = glob.glob(os.path.join(LOG_DIR, "Round_*"))
RUN_NUMBER = len(run_folders) + 1
CURRENT_RUN_NAME = f"Round_{RUN_NUMBER}"

# --- GESTIONE STATI CUSTOM ---
CUSTOM_STATES_DIR = "./train_states/"
CUSTOM_CAPTURE_STATES_DIR = "./captured_states/"
SAVE_PATH = "./models/"
os.makedirs(CUSTOM_STATES_DIR, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

# Lista file .state
STATE_FILES = glob.glob(os.path.join(CUSTOM_STATES_DIR, "*.state"))
STATES = [os.path.splitext(os.path.basename(f))[0] for f in STATE_FILES]

if len(STATES) == 0:
    STATES = [STATE_NAME]
    print(f"⚠️ train_states/ vuota. Uso default.")
else:
    print(f"✅ RUN {RUN_NUMBER} | Caricati {len(STATES)} stati per l'addestramento.")

NUM_ENVS = 12