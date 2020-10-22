import logging
import os
from collections import namedtuple
from typing import Set, List

from effects import get_effect, DESC_TO_PARAM, PARAM_TO_EFFECT, PARAM_TO_DESC
from util import parse_save_name

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

ProcessConfig = namedtuple(
    'ProcessConfig',
    'sr n_fft hop_length n_mels max_n_of_frames norm_audio norm_mel fmin fmax '
    'db_ref top_db n_mfcc calc_cent calc_bw calc_flat use_hashes proc_dir_name '
    'root_dir'
)


def create_save_dir(pc: ProcessConfig,
                    create_dirs: bool = True) -> str:
    assert os.path.exists(pc.root_dir)
    save_dir_name = create_save_name(pc)

    save_dir = os.path.join(pc.root_dir, pc.proc_dir_name)
    if not create_dirs:
        assert os.path.exists(save_dir)

    if not os.path.exists(save_dir):
        log.info('Creating processing folder.')
        os.makedirs(save_dir)

    save_dir = os.path.join(save_dir, save_dir_name)
    if not create_dirs:
        assert os.path.exists(save_dir)

    if not os.path.exists(save_dir):
        log.info(f'Creating dir: {save_dir_name}')
        os.makedirs(save_dir)

    return save_dir


def create_save_name(pc: ProcessConfig) -> str:
    save_name = f'proc__' \
                f'sr_{pc.sr}__' \
                f'n_fft_{pc.n_fft}__' \
                f'hop_length_{pc.hop_length}__' \
                f'n_mels_{pc.n_mels}__' \
                f'max_n_of_frames_{pc.max_n_of_frames}__' \
                f'norm_audio_{pc.norm_audio}__' \
                f'norm_mel_{pc.norm_mel}__' \
                f'fmin_{pc.fmin}__' \
                f'fmax_{pc.fmax}__' \
                f'db_ref_{pc.db_ref}__' \
                f'top_db_{pc.top_db}__' \
                f'n_mfcc_{pc.n_mfcc}__' \
                f'calc_cent_{pc.calc_cent}__' \
                f'calc_bw_{pc.calc_bw}__' \
                f'calc_flat_{pc.calc_flat}'

    return save_name


def generate_base_render_hash(orig_render_name: str,
                              base_effect_names: List[str],
                              exclude_descs: Set[str]) -> str:
    hash_tokens = ['_'.join(base_effect_names)]

    for param_str in orig_render_name.split('__'):
        if param_str.endswith('.wav'):
            param_str = param_str[:-4]
        split_param = param_str.split('_')
        desc = '_'.join(split_param[:-1])
        if desc in DESC_TO_PARAM:
            param = DESC_TO_PARAM[desc]
            effect = PARAM_TO_EFFECT[param]
            if effect.name in base_effect_names and desc not in exclude_descs:
                hash_tokens.append(param_str)

    render_hash = '__'.join(hash_tokens)

    if not render_hash:
        render_hash = 'dry'

    render_hash = f'{render_hash}.wav'
    return render_hash


def get_base_effect_info(orig_effect_dir_name: str,
                         exclude_descs: Set[str] = None) -> (str, List[str]):
    if exclude_descs is None:
        exclude_descs = set()

    orig_effect_dir_info = parse_save_name(orig_effect_dir_name, is_dir=True)
    gran = orig_effect_dir_info['gran']
    orig_effect_names = orig_effect_dir_info['name'].split('_')

    base_effect_names = []
    for effect_name in orig_effect_names:
        effect = get_effect(effect_name)
        if not all(PARAM_TO_DESC[p] in exclude_descs for p in effect.order):
            base_effect_names.append(effect_name)
    if not base_effect_names:
        base_effect_names.append('dry')

    base_effect_names = sorted(list(set(base_effect_names)))
    base_effect_dir_name = f'{"_".join(base_effect_names)}__gran_{gran}'
    return base_effect_dir_name, base_effect_names
