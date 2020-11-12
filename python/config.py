import logging
import os

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

# Paths
ROOT_DIR = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUT_DIR = os.path.join(ROOT_DIR, 'out')

CONFIGS_DIR = os.path.join(DATA_DIR, 'configs')
PRESETS_DIR = os.path.join(DATA_DIR, 'presets')
VST_DIR = os.path.join(DATA_DIR, 'vst')
DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')

EFFECTS_DIR = os.path.join(CONFIGS_DIR, 'effects')
DEFAULT_SERUM_PRESET_PATH = os.path.join(PRESETS_DIR, 'saw.fxp')

# Renderman
RM_SR = 44100
RM_BUFFER_SIZE = 512
RM_FFT_SIZE = 512

# Audio generation
RANDOM_GEN_THRESHOLD = 5
MAX_DUPLICATES = 50

# Mel spec
MEL_SR = RM_SR
HOP_LENGTH = 512
N_MELS = 128
N_FFT = 2048


def get_serum_path(instance: int = 1) -> str:
    serum_path = os.path.join(VST_DIR, f'Serum_{instance}.vst')
    assert os.path.exists(serum_path)
    return serum_path


SERUM_PATH = get_serum_path()
log.info(f'Serum path: {SERUM_PATH}')
