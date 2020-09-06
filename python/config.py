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

# Audio generation
RANDOM_GEN_THRESHOLD = 100
MAX_DUPLICATES = 50

# Mel spec
MEL_SR = RM_SR
HOP_LENGTH = 256
N_MELS = 256
N_FFT = 4096
MEL_MAX_DUR = 3.0
