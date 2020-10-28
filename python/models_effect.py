import logging
import os
from typing import List, Any, Callable, Tuple

import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, \
    Conv2D, Dropout, Bidirectional, Permute, Lambda, LSTM, Concatenate, \
    BatchNormalization
from tensorflow.python.framework.ops import Tensor

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def exposure_cnn(in_x: int,
                 in_y: int,
                 fc_dim: int = 128,
                 dropout: float = 0.5) -> (Tensor, Tensor):
    log.info(f'Using fc_dim of {fc_dim}')
    log.info(f'Using dropout of {dropout}')

    input_img = Input(shape=(in_x, in_y, 2))
    x = Conv2D(32,
               (4, 4),
               strides=(2, 2),
               padding='same',
               activation='elu')(input_img)
    x = Conv2D(64,
               (4, 4),
               strides=(2, 2),
               padding='same',
               activation='elu')(x)
    x = Conv2D(256,
               (4, 4),
               strides=(2, 2),
               padding='same',
               activation='elu')(x)
    x = Conv2D(256,
               (4, 4),
               strides=(2, 2),
               padding='same',
               activation='elu')(x)
    x = Flatten()(x)
    x = Dense(fc_dim, activation='elu')(x)
    fc = Dropout(dropout)(x)

    return input_img, fc


def baseline_cnn(in_x: int,
                 in_y: int,
                 n_channels: int = 2,
                 fc_dim: int = 128,
                 dropout: float = 0.5) -> (Tensor, Tensor):
    log.info(f'Using fc_dim of {fc_dim}')
    log.info(f'Using dropout of {dropout}')

    input_img = Input(shape=(in_x, in_y, n_channels))
    x = Conv2D(32,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(input_img)
    x = MaxPooling2D((4, 4))(x)
    x = Conv2D(64,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((4, 4))(x)
    x = Conv2D(64,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((4, 4))(x)
    x = Flatten()(x)
    x = Dense(fc_dim, activation='elu')(x)
    fc = Dropout(dropout)(x)

    return input_img, fc


def baseline_cnn_2x(in_x: int,
                    in_y: int,
                    n_mfcc: int = 30,
                    n_channels: int = 2,
                    fc_dim: int = 128,
                    dropout: float = 0.5,
                    use_fast_data_gen: bool = True) -> (Tensor, Tensor):
    assert n_mfcc > 0
    assert n_channels == 2
    log.info(f'Using fc_dim of {fc_dim}')
    log.info(f'Using dropout of {dropout}')

    if use_fast_data_gen:
        mels = Input(shape=(in_x, in_y), name='mels')
        mels_x = Lambda(lambda t: K.expand_dims(t, axis=-1))(mels)
        base_mels = Input(shape=(in_x, in_y), name='base_mels')
        base_mels_x = Lambda(lambda t: K.expand_dims(t, axis=-1))(base_mels)
        mel_input = Concatenate(axis=-1,
                                name='mel_input')([mels_x, base_mels_x])

        mfccs = Input(shape=(n_mfcc, in_y), name='mfccs')
        mfccs_x = Lambda(lambda t: K.expand_dims(t, axis=-1))(mfccs)
        base_mfccs = Input(shape=(n_mfcc, in_y), name='base_mfccs')
        base_mfccs_x = Lambda(lambda t: K.expand_dims(t, axis=-1))(base_mfccs)
        mfcc_input = Concatenate(axis=-1,
                                 name='mfcc_input')([mfccs_x, base_mfccs_x])
    else:
        mels = None
        base_mels = None
        mfccs = None
        base_mfccs = None
        mel_input = Input(shape=(in_x, in_y, n_channels), name='mel_input')
        mfcc_input = Input(shape=(n_mfcc, in_y, n_channels), name='mfcc_input')

    x = Conv2D(64,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(mel_input)
    x = MaxPooling2D((4, 4))(x)
    x = Conv2D(128,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((4, 4))(x)
    x = Conv2D(128,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((4, 4))(x)
    mel_x = Flatten(name='mel_flatten')(x)

    x = BatchNormalization(axis=-1)(mfcc_input)
    x = Conv2D(64,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((2, 4))(x)
    x = Conv2D(128,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((2, 4))(x)
    x = Conv2D(128,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((2, 4))(x)
    mfcc_x = Flatten(name='mfcc_flatten')(x)

    x = Concatenate(axis=-1)([mel_x, mfcc_x])
    x = Dense(fc_dim, activation='elu', name='fc_1')(x)
    x = Dropout(dropout)(x)
    x = Dense(fc_dim, activation='elu', name='fc_2')(x)
    fc = Dropout(dropout)(x)

    if use_fast_data_gen:
        return [mels, base_mels, mfccs, base_mfccs], fc
    else:
        return [mel_input, mfcc_input], fc


def baseline_cnn_shallow(in_x: int,
                         in_y: int,
                         n_channels: int = 2,
                         fc_dim: int = 128,
                         dropout: float = 0.5) -> (Tensor, Tensor):
    log.info(f'Using fc_dim of {fc_dim}')
    log.info(f'Using dropout of {dropout}')

    input_img = Input(shape=(in_x, in_y, n_channels))
    x = Conv2D(256,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(input_img)
    x = MaxPooling2D((4, 4))(x)
    x = Conv2D(128,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((4, 4))(x)
    x = Flatten()(x)
    x = Dense(2 * fc_dim, activation='elu')(x)
    x = Dropout(dropout)(x)
    x = Dense(fc_dim, activation='elu')(x)
    fc = Dropout(dropout)(x)

    return input_img, fc


def baseline_lstm(in_x: int,
                 in_y: int,
                 fc_dim: int = 128,
                 dropout: float = 0.5) -> (Tensor, Tensor):
    log.info(f'Using fc_dim of {fc_dim}')
    log.info(f'Using dropout of {dropout}')

    input_img = Input(shape=(in_x, in_y, 2))
    mel = Lambda(lambda t: t[:, :, :, 0])(input_img)
    base_mel = Lambda(lambda t: t[:, :, :, 1])(input_img)

    x = Permute((2, 1))(mel)
    x = Bidirectional(LSTM(fc_dim))(x)
    x = Dense(fc_dim, activation='elu')(x)
    x_mel = Dropout(dropout)(x)

    x = Permute((2, 1))(base_mel)
    x = Bidirectional(LSTM(fc_dim))(x)
    x = Dense(fc_dim, activation='elu')(x)
    x_base_mel = Dropout(dropout)(x)

    fc = Concatenate()([x_mel, x_base_mel])

    return input_img, fc


def cnn_input_adapter(in_x: int,
                      in_y: int,
                      n_mfcc: int) -> Tuple[Tensor, ...]:
    mels = Input(shape=(in_x, in_y), name='mels')
    mels_x = Lambda(lambda t: K.expand_dims(t, axis=-1))(mels)
    base_mels = Input(shape=(in_x, in_y), name='base_mels')
    base_mels_x = Lambda(lambda t: K.expand_dims(t, axis=-1))(base_mels)
    mel_input = Concatenate(axis=-1, name='mel_input')([mels_x, base_mels_x])

    mfccs = Input(shape=(n_mfcc, in_y), name='mfccs')
    mfccs_x = Lambda(lambda t: K.expand_dims(t, axis=-1))(mfccs)
    base_mfccs = Input(shape=(n_mfcc, in_y), name='base_mfccs')
    base_mfccs_x = Lambda(lambda t: K.expand_dims(t, axis=-1))(base_mfccs)
    mfcc_input = Concatenate(axis=-1,
                             name='mfcc_input')([mfccs_x, base_mfccs_x])

    return mels, base_mels, mfccs, base_mfccs, mel_input, mfcc_input


def build_effect_model(in_x: int,
                       in_y: int,
                       n_mfcc: int = 30,
                       architecture: Callable = baseline_cnn_2x,
                       n_bin: int = 0,
                       n_cate: List[int] = None,
                       cate_names: List[str] = None,
                       n_cont: int = 0,
                       **kwargs: Any) -> Model:
    if cate_names is None:
        cate_names = []
    if n_cate is None:
        n_cate = []
    log.info(f'Using architecture: {architecture.__name__}')
    cnn_inputs, fc = architecture(in_x, in_y, n_mfcc=n_mfcc, **kwargs)

    cnn_outputs = []
    if n_bin:
        bin_output = Dense(n_bin, activation='sigmoid', name='bin_output')(fc)
        cnn_outputs.append(bin_output)

    for n_classes, name in zip(n_cate, cate_names):
        cate_output = Dense(n_classes, activation='softmax', name=name)(fc)
        cnn_outputs.append(cate_output)

    if n_cont:
        cont_output = Dense(n_cont, activation='relu', name='cont_output')(fc)
        cnn_outputs.append(cont_output)

    assert cnn_outputs
    cnn_model = Model(cnn_inputs, cnn_outputs, name=architecture.__name__)

    # mels, base_mels, mfccs, base_mfccs, mel_input, mfcc_input = cnn_input_adapter(in_x, in_y, n_mfcc)
    # outputs = cnn_model([mel_input, mfcc_input])
    # model = Model([mels, base_mels, mfccs, base_mfccs], outputs)
    # return model
    return cnn_model
