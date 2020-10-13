import copy
import logging
import os
from itertools import combinations
from typing import Set

import yaml
from tqdm import tqdm

from audio_rendering_util import RenderConfig, PatchGenerator, create_save_dir, \
    generate_render_hash, render_patch
from config import CONFIGS_DIR, RANDOM_GEN_THRESHOLD, MAX_DUPLICATES
from effects import DESC_TO_PARAM, get_effect, PARAM_TO_EFFECT
from serum_util import setup_serum, set_preset
from util import get_render_names, generate_exclude_descs, parse_save_name

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def render_audio(rc: RenderConfig,
                 max_duplicates_in_a_row: int = MAX_DUPLICATES) -> None:
    pg = PatchGenerator(rc.gran, rc.effects)
    assert rc.effect_names() == pg.effect_names
    save_dir = create_save_dir(rc, create_dirs=True)

    render_names = get_render_names(save_dir,
                                    assert_unique=True,
                                    use_hashes=rc.use_hashes)
    n_existing_renders = len(render_names)
    log.info(f'{n_existing_renders} existing renders found.')

    engine = setup_serum(rc.preset, sr=rc.sr, render_once=True)

    log.info(f'{pg.n_combos} no. of possible rendering combos.')
    log.info(f'{pg.effect_names} effects in patch generator.')

    for effect_name in pg.effect_names:
        effect = get_effect(effect_name)
        log.info(f'Setting default {effect.name} params.')
        set_preset(engine, effect.default)

    if rc.n > 0 and RANDOM_GEN_THRESHOLD * rc.n > pg.n_combos:
        log.warning(f'n of {rc.n} is too big, so generating all combos.')
        n_to_render = -1
    else:
        n_to_render = rc.n

    if n_to_render > 0 and 0 < rc.max_n < n_existing_renders + n_to_render:
        n_to_render = max(0, rc.max_n - n_existing_renders)
        log.info(f'n reduced to {n_to_render} due to max n of {rc.max_n}')

    # TODO
    for exclude_dir in rc.exclude_dirs:
        log.info(f'Excluding renders in {exclude_dir}')
        n_existing_renders = len(render_names)
        for render_name in os.listdir(exclude_dir):
            if render_name.endswith('.wav'):
                render_names.add(render_name)

        n_increase = len(render_names) - n_existing_renders
        if n_increase > 0:
            log.info(f'Existing renders increased by {n_increase} to '
                     f'{len(render_names)}')

    if n_to_render == -1:
        pbar = tqdm(total=pg.n_combos)

        for default_diff, patch in pg.generate_all_combos():
            render_name = generate_render_hash(pg.effect_names,
                                               default_diff,
                                               pg.param_n_digits)

            if render_name in render_names:
                log.debug(f'Duplicate render generated: {render_name}')
            else:
                render_patch(engine,
                             patch,
                             rc,
                             save_dir,
                             render_name)

                render_names.add(render_name)

            pbar.update(1)
    else:
        n_rendered = 0
        duplicates_in_a_row = 0
        pbar = tqdm(total=n_to_render)
        while n_rendered < n_to_render:
            default_diff, patch = pg.generate_random_patch()
            render_name = generate_render_hash(pg.effect_names,
                                               default_diff,
                                               pg.param_n_digits)

            if render_name in render_names:
                duplicates_in_a_row += 1
                log.debug(f'Duplicate render generated: {render_name}')

                if duplicates_in_a_row > max_duplicates_in_a_row:
                    log.warning('Too many duplicates generated in a row.')
                    break
            else:
                render_patch(engine,
                             patch,
                             rc,
                             save_dir,
                             render_name)

                render_names.add(render_name)
                n_rendered += 1
                duplicates_in_a_row = 0
                pbar.update(1)


def render_base_audio(orig_rc: RenderConfig,
                      exclude_effects: Set[str] = None,
                      exclude_params: Set[str] = None,
                      use_hashes: bool = False) -> None:
    if exclude_effects is None:
        exclude_effects = set()
    if exclude_params is None:
        exclude_params = set()
    exclude_descs = generate_exclude_descs(exclude_effects, exclude_params)
    log.info(f'Exclude effects = {exclude_effects}')
    log.info(f'Exclude params = {exclude_params}')
    log.info(f'Exclude descs = {exclude_descs}')

    orig_save_dir = create_save_dir(orig_rc, create_dirs=False)
    log.info(f'Original save dir = {orig_save_dir}')

    orig_render_names = get_render_names(orig_save_dir,
                                         assert_unique=True,
                                         use_hashes=orig_rc.use_hashes)
    log.info(f'{len(orig_render_names)} existing original renders found.')

    rc = RenderConfig(preset=orig_rc.preset,
                      sr=orig_rc.sr,
                      note_length=orig_rc.note_length,
                      render_length=orig_rc.render_length,
                      midi=orig_rc.midi,
                      vel=orig_rc.vel,
                      gran=orig_rc.gran,
                      n=orig_rc.n,
                      max_n=orig_rc.max_n,
                      root_dir=orig_rc.root_dir,
                      effects=copy.deepcopy(orig_rc.effects),
                      use_hashes=use_hashes)

    for desc in exclude_descs:
        for effect in rc.effects:
            if desc in effect:
                del effect[desc]

    rc.effects = list(filter(lambda e: len(e) > 1, rc.effects))
    base_effect_names = rc.effect_names()
    log.info(f'{base_effect_names} effects in base render config.')

    log.info(f'Creating base renders directory.')
    save_dir = create_save_dir(rc, create_dirs=True)
    log.info(f'Base save dir = {save_dir}')

    base_render_names = get_render_names(save_dir,
                                         assert_unique=True,
                                         use_hashes=rc.use_hashes)
    n_existing_renders = len(base_render_names)
    log.info(f'{n_existing_renders} existing base renders found.')

    engine = setup_serum(rc.preset, sr=rc.sr, render_once=True)

    for effect_name in base_effect_names:
        effect = get_effect(effect_name)
        log.info(f'Setting default {effect.name} params.')
        set_preset(engine, effect.default)

    for orig_render_name in tqdm(orig_render_names):
        render_info = parse_save_name(orig_render_name, is_dir=False)
        rc_effects = {}
        for desc, param_v in render_info.items():
            if desc != 'name' and desc not in exclude_descs:
                param = DESC_TO_PARAM[desc]
                effect_name = PARAM_TO_EFFECT[param].name
                if effect_name not in rc_effects:
                    rc_effects[effect_name] = {'name': effect_name}
                rc_effect = rc_effects[effect_name]
                rc_effect[desc] = [param_v]

        rc.effects = list(rc_effects.values())
        pg = PatchGenerator(rc.gran, rc.effects)
        assert pg.n_combos == 1
        assert rc.effect_names() == pg.effect_names

        # Using base effect names is important due to render naming convention
        for effect_name in base_effect_names:
            effect = get_effect(effect_name)
            set_preset(engine, effect.default)

        default_diff, patch = list(pg.generate_all_combos())[0]
        render_name = generate_render_hash(base_effect_names,
                                           default_diff,
                                           pg.param_n_digits)

        if render_name in base_render_names:
            log.debug(f'Duplicate base render generated: {render_name}')
        else:
            render_patch(engine,
                         patch,
                         rc,
                         save_dir,
                         render_name)

            base_render_names.add(render_name)

    log.info(f'{len(base_render_names) - n_existing_renders} new base '
             f'renders rendered.')


if __name__ == '__main__':
    # render_config_path = os.path.join(CONFIGS_DIR, 'rendering/seq_5_v3_train.yaml')
    # with open(render_config_path, 'r') as config_f:
    #     render_config = yaml.full_load(config_f)
    # rc = RenderConfig(**render_config)
    # render_audio(rc)
    # exit()

    all_effects = ['compressor', 'distortion', 'eq', 'phaser', 'reverb-hall']
    all_combos = []
    for n_effects in range(len(all_effects) + 1):
        for combo in combinations(all_effects, n_effects):
            all_combos.append(set(list(combo)))

    all_combos.reverse()

    log.info(f'All exclude combos = {all_combos}')
    log.info(f'Len of exclude combos = {len(all_combos)}')

    render_config_path = os.path.join(CONFIGS_DIR, 'rendering/seq_5_v3_train.yaml')
    with open(render_config_path, 'r') as config_f:
        render_config = yaml.full_load(config_f)

    orig_rc = RenderConfig(**render_config)

    for combo in all_combos:
        if len(combo) > 2:  # TODO
            use_hashes = False
        else:
            use_hashes = True
        exclude_effects = set(combo)
        render_base_audio(orig_rc,
                          exclude_effects=exclude_effects,
                          use_hashes=use_hashes)
    exit()
