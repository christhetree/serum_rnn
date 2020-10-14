import logging
import os
from typing import List, Set, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from config import OUT_DIR, DATASETS_DIR
from effects import DESC_TO_PARAM, PARAM_TO_EFFECT
from models import build_effect_model, baseline_cnn_2x
from training_util import TestDataGenerator, DataGenerator, XYMetaData
from util import get_effect_names

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

GPU = 0
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    log.info(f'GPUs available: {physical_devices}')
    tf.config.experimental.set_visible_devices(physical_devices[GPU], 'GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[GPU],
    #                                          enable=True)


def get_eval_cnn_spec(gen: TestDataGenerator,
                      save_name: str,
                      max_n: int = 1000) -> None:
    x_s = []
    render_names = []
    base_render_names = []
    presets = []
    for (x, render_name, base_render_name, preset), y in tqdm(gen):
        if len(render_names) >= max_n:
            break

        effect_names = get_effect_names(render_name)
        base_effect_names = get_effect_names(base_render_name)
        if len(base_effect_names) + 1 == len(effect_names):
            x_s.append(x)
            render_names.append(render_name)
            base_render_names.append(base_render_name)
            presets.append(preset)

    assert len(x_s) == len(render_names) == len(base_render_names) == len(presets)
    log.info(f'Length of x_s = {len(x_s)}')
    log.info(f'Saving: {save_name}')
    save_path = os.path.join(OUT_DIR, save_name)
    log.info('converting')
    x_s = np.concatenate(x_s, axis=0)
    log.info(x_s.shape)
    log.info('converting done')
    np.savez(save_path,
             x_s=x_s,
             render_names=render_names,
             base_render_names=base_render_names,
             presets=presets)


def train_model_gen(model: Model,
                    train_gen: DataGenerator,
                    val_gen: DataGenerator,
                    model_name: str,
                    epochs: int = 100,
                    patience: int = 8,
                    output_dir_path: str = OUT_DIR,
                    use_multiprocessing: bool = True,
                    workers: int = 8) -> None:
    save_path = os.path.join(
        output_dir_path,
        # model_name + '_e{epoch:03d}_vl{val_loss:.4f}.h5'
        f'{model_name}__best.h5'
    )
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=patience,
                       verbose=1)
    cp = ModelCheckpoint(save_path,
                         monitor='val_loss',
                         save_best_only=True,
                         verbose=1)
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=epochs,
              callbacks=[es, cp],
              use_multiprocessing=use_multiprocessing,
              workers=workers,
              verbose=1)


def get_x_y_metadata(data_dir: str,
                     y_params: Set[int]) -> XYMetaData:
    assert os.path.exists(data_dir)
    x_dir = os.path.join(data_dir, 'x')
    assert os.path.exists(x_dir)

    sample_x_name = None
    sample_x_data = None
    for npz_name in os.listdir(x_dir):
        if npz_name.endswith('.npz'):
            sample_x_name = npz_name
            sample_x_data = np.load(os.path.join(x_dir, sample_x_name))
            break
    assert sample_x_name
    assert sample_x_data is not None

    sample_mel_path = sample_x_data['mel_path'].item()
    assert os.path.exists(sample_mel_path)
    sample_mel = np.load(sample_mel_path)['mel']

    log.info(f'Input spectrogram shape = {sample_mel.shape}')
    assert len(sample_mel.shape) == 2
    in_x = sample_mel.shape[0]
    in_y = sample_mel.shape[1]

    sample_base_mel_path = sample_x_data['base_mel_path'].item()
    assert os.path.exists(sample_base_mel_path)
    sample_base_mel = np.load(sample_base_mel_path)['mel']
    assert sample_mel.shape == sample_base_mel.shape

    y_dir = os.path.join(data_dir, 'y')
    assert os.path.exists(y_dir)

    y_params = sorted(list(y_params))
    y_params_str = '_'.join(str(p) for p in y_params)

    sample_y_data = np.load(
        os.path.join(y_dir, f'{sample_x_name}__y_{y_params_str}.npz'))
    n_bin = 0
    n_cont = 0
    descs = []

    for key, values in sample_y_data.items():
        if key == 'binary':
            n_bin = len(values)
        elif key == 'continuous':
            n_cont = len(values)
        else:
            descs.append(key)

    descs = sorted(descs)
    log.info(f'n_bin = {n_bin}')
    log.info(f'descs = {descs}')
    log.info(f'n_cont = {n_cont}')

    n_cate = []
    cate_names = []
    y_losses = {}
    metrics = {}

    if n_bin:
        y_losses['bin_output'] = 'bce'
        metrics['bin_output'] = 'acc'

    for desc in descs:
        param = DESC_TO_PARAM[desc]
        effect = PARAM_TO_EFFECT[param]
        n_cate.append(effect.categorical[param])

        cate_name = desc.strip().lower().replace(' ', '_')
        cate_names.append(cate_name)

        y_losses[cate_name] = 'sparse_categorical_crossentropy'
        metrics[cate_name] = 'acc'

    if n_cont:
        y_losses['cont_output'] = 'mse'
        metrics['cont_output'] = 'mae'

    log.info(f'n_cate = {n_cate}')
    log.info(f'cate_names = {cate_names}')
    log.info(f'y_losses = {y_losses}')
    log.info(f'metrics = {metrics}')

    x_y_metadata = XYMetaData(data_dir=data_dir,
                              x_dir=x_dir,
                              in_x=in_x,
                              in_y=in_y,
                              y_dir=y_dir,
                              y_params=set(y_params),
                              y_params_str=y_params_str,
                              n_bin=n_bin,
                              n_cate=n_cate,
                              n_cont=n_cont,
                              descs=descs,
                              cate_names=cate_names,
                              y_losses=y_losses,
                              metrics=metrics)

    return x_y_metadata


def get_x_ids(data_dir: str,
              val_split: float = 0.10,
              test_split: float = 0.05,
              max_n: int = -1,
              use_cached: bool = True) -> (List[Tuple[str, str, str]],
                                           List[Tuple[str, str, str]],
                                           List[Tuple[str, str, str]]):
    assert val_split + test_split < 1.0
    train_x_ids_path = os.path.join(data_dir, 'train_x_ids.npy')
    val_x_ids_path = os.path.join(data_dir, 'val_x_ids.npy')
    test_x_ids_path = os.path.join(data_dir, 'test_x_ids.npy')

    if use_cached \
        and all(os.path.exists(p)
                for p in [train_x_ids_path, val_x_ids_path, test_x_ids_path]):
        log.info('Using cached x_ids.')
        train_x_ids = np.load(train_x_ids_path)
        val_x_ids = np.load(val_x_ids_path)
        test_x_ids = np.load(test_x_ids_path)
    else:
        log.info('Creating new x_ids.')
        x_dir = os.path.join(data_dir, 'x')
        x_ids = []
        for npz_name in tqdm(os.listdir(x_dir)):
            if not npz_name.endswith('.npz'):
                continue

            npz_data = np.load(os.path.join(x_dir, npz_name))
            mel_path = npz_data['mel_path'].item()
            base_mel_path = npz_data['base_mel_path'].item()
            x_ids.append((npz_name, mel_path, base_mel_path))

        log.info(f'Found {len(x_ids)} data points.')

        np.random.shuffle(x_ids)
        if max_n > 0:
            x_ids = x_ids[:max_n]

        val_idx = int(len(x_ids) * (1.0 - val_split - test_split))
        test_idx = int(len(x_ids) * (1.0 - test_split))

        train_x_ids = x_ids[:val_idx]
        val_x_ids = x_ids[val_idx:test_idx]
        test_x_ids = x_ids[test_idx:]

        log.info('Caching x_ids.')
        np.save(train_x_ids_path, train_x_ids)
        np.save(val_x_ids_path, val_x_ids)
        np.save(test_x_ids_path, test_x_ids)

    return train_x_ids, val_x_ids, test_x_ids


if __name__ == '__main__':
    # effect = 'compressor'
    # params = {270, 271, 272}
    # effect = 'distortion'
    # params = {97, 99}
    # effect = 'eq'
    # params = {89, 91, 93}
    # effect = 'phaser'
    # params = {112, 113, 114}
    effect = 'reverb-hall'
    params = {81, 84, 86}

    # architecture = baseline_cnn
    architecture = baseline_cnn_2x
    # architecture = baseline_cnn_shallow
    # architecture = exposure_cnn
    # architecture = baseline_lstm

    batch_size = 128
    epochs = 100
    val_split = 0.10
    test_split = 0.05
    patience = 8
    used_cached_x_ids = True
    max_n = -1
    channel_mode = 1
    use_multiprocessing = True
    workers = 8
    load_prev_model = False
    # load_prev_model = True

    # presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    presets_cat = 'temporal'

    # model_name = f'testing__{effect}__{architecture.__name__}__cm_{channel_mode}'
    model_name = f'seq_5_v3__{presets_cat}__{effect}__{architecture.__name__}' \
                 f'__cm_{channel_mode}'

    datasets_dir = DATASETS_DIR
    # data_dir = os.path.join(datasets_dir, f'testing__{effect}')
    data_dir = os.path.join(datasets_dir, f'seq_5_v3__{presets_cat}__{effect}')
    log.info(f'data_dir = {data_dir}')

    x_y_metadata = get_x_y_metadata(data_dir, params)
    train_x_ids, val_x_ids, test_x_ids = get_x_ids(data_dir,
                                                   val_split=val_split,
                                                   test_split=test_split,
                                                   max_n=max_n,
                                                   use_cached=used_cached_x_ids)
    log.info(f'train_x_ids length = {len(train_x_ids)}')
    log.info(f'val_x_ids length = {len(val_x_ids)}')
    log.info(f'test_x_ids length = {len(test_x_ids)}')
    log.info(f'batch_size = {batch_size}')

    # test_x_ids = test_x_ids[:100]
    spec_gen = TestDataGenerator(test_x_ids,
                                 x_y_metadata,
                                 batch_size=1,
                                 channel_mode=1,
                                 shuffle=True)
    save_name = f'{model_name}__eval_spec_data.npz'
    get_eval_cnn_spec(spec_gen, save_name)
    print('done!')
    exit()

    train_gen = DataGenerator(train_x_ids,
                              x_y_metadata,
                              batch_size=batch_size,
                              channel_mode=channel_mode)
    val_gen = DataGenerator(val_x_ids,
                            x_y_metadata,
                            batch_size=batch_size,
                            channel_mode=channel_mode)

    model = build_effect_model(x_y_metadata.in_x,
                               x_y_metadata.in_y,
                               architecture=architecture,
                               n_bin=x_y_metadata.n_bin,
                               n_cate=x_y_metadata.n_cate,
                               cate_names=x_y_metadata.cate_names,
                               n_cont=x_y_metadata.n_cont)

    if load_prev_model:
        log.info('Loading previous best model.')
        model.load_weights(os.path.join(OUT_DIR, f'{model_name}__best.h5'))

    model.compile(optimizer='adam',
                  loss=x_y_metadata.y_losses,
                  metrics=x_y_metadata.metrics)
    model.summary()

    train_model_gen(model,
                    train_gen,
                    val_gen,
                    model_name,
                    epochs=epochs,
                    patience=patience,
                    use_multiprocessing=use_multiprocessing,
                    workers=workers)
