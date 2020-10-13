import logging
import os
from typing import List, Dict

import yaml

from config import CONFIGS_DIR, EFFECTS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class Effect:
    def __init__(self,
                 name: str,
                 default: Dict[int, float],
                 binary: List[int],
                 categorical: Dict[int, int],
                 continuous: List[int]) -> None:
        super().__init__()
        assert name is not None
        self.name = name

        if default is None:
            self.default = {}
        else:
            assert isinstance(default, dict)
            self.default = default

        if binary is None:
            self.binary = set()
        else:
            assert isinstance(binary, list)
            self.binary = set(binary)

        if categorical is None:
            self.categorical = {}
        else:
            assert isinstance(categorical, dict)
            self.categorical = categorical

        if continuous is None:
            self.continuous = set()
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

        assert len(order) == len(set(order))
        assert all(p in self.default for p in order)
        self.order = order


EFFECTS = {}
for effect_config_name in os.listdir(EFFECTS_DIR):
    if effect_config_name.endswith('.yaml'):
        with open(os.path.join(EFFECTS_DIR, effect_config_name), 'r') as f:
            effect_config = yaml.full_load(f)
            effect = Effect(**effect_config)
            assert effect.name not in EFFECTS
            EFFECTS[effect.name] = effect

log.info(f'Supported effects: {sorted(list(EFFECTS.keys()))}')

with open(os.path.join(CONFIGS_DIR, 'serum_desc_to_param.yaml'), 'r') as f:
    DESC_TO_PARAM = yaml.full_load(f)

with open(os.path.join(CONFIGS_DIR, 'serum_param_to_desc.yaml'), 'r') as f:
    PARAM_TO_DESC = yaml.full_load(f)

assert all(p in PARAM_TO_DESC for p in DESC_TO_PARAM.values())
assert all(d in DESC_TO_PARAM for d in PARAM_TO_DESC.values())
assert 'name' not in DESC_TO_PARAM
assert all(effect_name not in DESC_TO_PARAM for effect_name in EFFECTS.keys())

ALL_PARAMS = [p for e in EFFECTS.values() for p in e.order]
assert len(ALL_PARAMS) == len(set(ALL_PARAMS))
assert all(p in PARAM_TO_DESC for p in ALL_PARAMS)

PARAM_TO_EFFECT = {p: e for e in EFFECTS.values() for p in e.order}
PARAM_TO_TYPE = {}
for effect in EFFECTS.values():
    for param in effect.binary:
        PARAM_TO_TYPE[param] = 'binary'
    for param in effect.categorical.keys():
        PARAM_TO_TYPE[param] = 'categorical'
    for param in effect.continuous:
        PARAM_TO_TYPE[param] = 'continuous'


def get_effect(name: str) -> Effect:
    return EFFECTS[name]
