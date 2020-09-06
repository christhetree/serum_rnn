import logging
import os
from typing import List, Dict

import yaml

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class Effect:
    def __init__(self,
                 name: str,
                 default: Dict[int, float],
                 binary: List[int],
                 categorical: Dict[int, int],
                 continuous: List[int]):
        super().__init__()
        self.name = name
        self.default = default
        self.binary = binary
        self.categorical = categorical
        self.continuous = continuous

        order = []
        if self.continuous is not None:
            order.extend(sorted(self.continuous))
        if self.categorical is not None:
            order.extend(sorted(list(self.categorical.keys())))
        if self.binary is not None:
            order.extend(sorted(self.binary))

        self.order = order


# Paths
ROOT_DIR = '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/'

OUT_DIR = os.path.join(ROOT_DIR, 'out')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

CONFIGS_DIR = os.path.join(DATA_DIR, 'configs')
PRESETS_DIR = os.path.join(DATA_DIR, 'presets')
VST_DIR = os.path.join(DATA_DIR, 'vst')
DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')

SERUM_PATH = os.path.join(VST_DIR, 'Serum.vst')
log.info(f'Serum path: {SERUM_PATH}')
DEFAULT_SERUM_PRESET_PATH = os.path.join(PRESETS_DIR, 'default.fxp')

with open(os.path.join(CONFIGS_DIR, 'serum_desc_to_param.yaml'), 'r') as f:
    DESC_TO_PARAM = yaml.full_load(f)

with open(os.path.join(CONFIGS_DIR, 'serum_param_to_desc.yaml'), 'r') as f:
    PARAM_TO_DESC = yaml.full_load(f)

with open(os.path.join(CONFIGS_DIR, 'distortion.yaml'), 'r') as f:
    distortion_config = yaml.full_load(f)

with open(os.path.join(CONFIGS_DIR, 'flanger.yaml'), 'r') as f:
    flanger_config = yaml.full_load(f)

distortion = Effect(**distortion_config)
flanger = Effect(**flanger_config)

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
