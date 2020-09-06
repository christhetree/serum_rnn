import logging
import os
from collections import namedtuple

import yaml

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)

# Paths
ROOT_DIR = '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/'

OUT_DIR = os.path.join(ROOT_DIR, 'out')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

CONFIGS_DIR = os.path.join(DATA_DIR, 'configs')
PRESETS_DIR = os.path.join(DATA_DIR, 'presets')
VST_DIR = os.path.join(DATA_DIR, 'vst')
DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')

SERUM_PATH = os.path.join(VST_DIR, 'Serum.vst')
DEFAULT_SERUM_PRESET_PATH = os.path.join(PRESETS_DIR, 'default.fxp')

with open(os.path.join(CONFIGS_DIR, 'serum_desc_to_param.yaml'), 'r') as f:
    DESC_TO_PARAM = yaml.full_load(f)

with open(os.path.join(CONFIGS_DIR, 'serum_param_to_desc.yaml'), 'r') as f:
    PARAM_TO_DESC = yaml.full_load(f)

with open(os.path.join(CONFIGS_DIR, 'distortion.yaml'), 'r') as f:
    distortion_config = yaml.full_load(f)

with open(os.path.join(CONFIGS_DIR, 'flanger.yaml'), 'r') as f:
    flanger_config = yaml.full_load(f)

Effect = namedtuple('Effect', 'name default binary categorical continuous')

distortion = Effect(**distortion_config)
flanger = Effect(**flanger_config)

log.info(f'Serum path: {SERUM_PATH}')

# Renderman
RM_SR = 44100
RM_BUFFER_SIZE = 512
RM_FFT_SIZE = 512

# # Audio rendering
NOTE_LENGTH = 1.0
RENDER_LENGTH = 1.0
PARAM_GRANULARITY = 100

# Mel spec
MEL_SR = RM_SR
HOP_LENGTH = 256
N_MELS = 256
N_FFT = 4096
MEL_MAX_DUR = 3.0

# Default param values
DEFAULT_DISTORTION = {
    96: 1.0,   # Dist_Wet
    97: 0.25,  # Dist_Drv
    98: 0.0,   # Dist_L/B/H
    99: 0.0,   # Dist_Mode
    100: 0.5,  # Dist_Freq
    101: 0.5,  # Dist_BW
    102: 0.0,  # Dist_PrePost
    154: 1.0,  # Dist Enable
}

DEFAULT_FLANGER = {
    103: 1.0,  # Flg_Wet
    104: 0.0,  # Flg_BPM_Sync
    105: 0.25,  # Flg_Rate
    106: 1.0,  # Flg_Dep
    107: 0.5,  # Flg_Feed
    108: 0.5,  # Flg_Stereo
    155: 1.0,  # Flg Enable
}
