import logging
import os
from typing import List, Dict

import yaml

from python.config import CONFIGS_DIR

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
        assert name is not None
        self.name = name

        if default is None:
            self.default = {}
        else:
            assert isinstance(default, dict)
            self.default = default

        if binary is None:
            self.binary = set
        else:
            assert isinstance(binary, list)
            self.binary = set(binary)

        if categorical is None:
            self.categorical = {}
        else:
            assert isinstance(categorical, dict)
            self.categorical = categorical

        if continuous is None:
            self.continuous = set
        else:
            assert isinstance(continuous, list)
            self.continuous = set(continuous)

        order = []
        if self.continuous is not None:
            order.extend(sorted(self.continuous))
        if self.categorical is not None:
            order.extend(sorted(list(self.categorical.keys())))
        if self.binary is not None:
            order.extend(sorted(self.binary))

        self.order = order


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

effects = {
    'distortion': distortion,
    'flanger': flanger,
}
log.info(f'Supported effects: {sorted(list(effects.keys()))}')

param_to_effect = {p: e for e in effects.values() for p in e.order}
param_to_type = {}
for effect in effects.values():
    for param in effect.binary:
        param_to_type[param] = 'binary'
    for param in effect.categorical.keys():
        param_to_type[param] = 'categorical'
    for param in effect.continuous:
        param_to_type[param] = 'continuous'


def get_effect(name: str) -> Effect:
    return effects[name]
