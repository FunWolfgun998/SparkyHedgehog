import os
import glob
import stable_retro as retro

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

GAME_NAME = 'SonicTheHedgehog-Genesis-v0'
STATE_NAME = 'GreenHillZone.Act1'

LOG_DIR = os.path.join(ROOT_DIR, "logs/Sparky 4_0")
VIDEO_DIR = os.path.join(ROOT_DIR, "utils/video_models")
CUSTOM_STATES_DIR = os.path.join(ROOT_DIR, "train_states")
CUSTOM_CAPTURE_STATES_DIR = os.path.join(ROOT_DIR, "utils/captured_states")
SAVE_PATH = os.path.join(ROOT_DIR, "models/Sparky 4_0")
SAVE_TEXT_LOGS = True

# --- AUTOMAZIONE RUN (TensorBoard Continuity) ---
os.makedirs(LOG_DIR, exist_ok=True)
run_folders = glob.glob(os.path.join(LOG_DIR, "Round_*"))
RUN_NUMBER = len(run_folders) + 1
CURRENT_RUN_NAME = f"Round_{RUN_NUMBER}"

# --- GESTIONE STATI ---
def get_state_files():
    """Ritorna i percorsi completi dei file .state."""
    return glob.glob(os.path.join(CUSTOM_STATES_DIR, "*.state"))

STATE_FILES = get_state_files()
STATES = [os.path.splitext(os.path.basename(f))[0] for f in STATE_FILES]
if len(STATES) == 0:
    STATES = [STATE_NAME]

NUM_ENVS = 30