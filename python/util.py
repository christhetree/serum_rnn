import logging
import os
from typing import Dict, Union, Set

from effects import get_effect, PARAM_TO_DESC

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def parse_save_name(save_name: str,
                    is_dir: bool = False) -> Dict[str, Union[int, float, bool]]:
    if not is_dir:
        save_name = os.path.splitext(save_name.strip())[0]  # Delete file ext.
    tokens = save_name.split('__')
    name = tokens[0]
    result = {'name': name}

    for token in tokens[1:]:
        sub_tokens = token.split('_')
        if len(sub_tokens) > 1:
            key = '_'.join(sub_tokens[:-1])
            value = sub_tokens[-1]
            if '.' in value:
                result[key] = float(value)
            elif value == 'T':
                result[key] = True
            elif value == 'F':
                result[key] = False
            else:
                result[key] = int(value)
        else:
            result[token] = True

    return result


def get_render_names(renders_dir: str,
                     assert_unique: bool = True,
                     use_hashes: bool = False) -> Set[str]:
    render_names = set()
    if use_hashes:
        mapping_path = os.path.join(renders_dir, 'mapping.txt')
        if not os.path.exists(mapping_path):
            log.info('Creating mapping file.')
            with open(mapping_path, 'w') as _:
                pass
        with open(os.path.join(mapping_path), 'r') as mapping_f:
            for line in mapping_f:
                render_hash, render_name = line.strip().split('\t')
                if assert_unique:
                    assert render_name not in render_names
                render_names.add(render_name)
    else:
        for render_name in os.listdir(renders_dir):
            if render_name.endswith('.wav'):
                if assert_unique:
                    assert render_name not in render_names
                render_names.add(render_name)
    return render_names


def get_mapping(mapping_path: str) -> Dict[str, str]:
    mapping = {}
    with open(os.path.join(mapping_path), 'r') as mapping_f:
        for line in mapping_f:
            render_hash, render_name = line.strip().split('\t')
            mapping[render_name] = render_hash
    return mapping


def generate_exclude_descs(exclude_effects: Set[str],
                           exclude_params: Set[int]) -> Set[str]:
    exclude_descs = set()
    all_exclude_params = set(exclude_params)
    for effect_name in exclude_effects:
        effect = get_effect(effect_name)
        for param in effect.order:
            all_exclude_params.add(param)
            desc = PARAM_TO_DESC[param]
            exclude_descs.add(desc)

    for param in all_exclude_params:
        desc = PARAM_TO_DESC[param]
        exclude_descs.add(desc)

    return exclude_descs
