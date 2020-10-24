import logging
import os
from typing import Callable, Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, \
    Conv2D, Dropout, Bidirectional, Lambda, LSTM, Concatenate, \
    TimeDistributed, Masking, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.framework.ops import Tensor

from models_effect import baseline_cnn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def baseline_effect_rnn(in_x: int = 128,
                        in_y: int = 88,
                        n_channels: int = 2,
                        n_effects: int = 5,
                        lstm_dim: int = 128,
                        fc_dim: int = 128,
                        dropout: float = 0.5,
                        cnn_architecture: Callable = baseline_cnn,
                        cnn_fc_dim: int = 128,
                        cnn_dropout: float = 0.5) -> Model:
    cnn_input_img, cnn_fc = cnn_architecture(in_x,
                                             in_y,
                                             fc_dim=cnn_fc_dim,
                                             n_channels=n_channels,
                                             dropout=cnn_dropout)
    cnn = Model(cnn_input_img, cnn_fc, name='cnn')

    img_seq = Input(shape=(None, in_x, in_y, n_channels), name='img_seq')
    x = Masking(mask_value=0.0, name='img_seq_mask')(img_seq)
    img_emb = TimeDistributed(cnn, name='img_emb')(x)

    effect_seq = Input(shape=(None, n_effects + 1), name='effect_seq')
    effect_emb = Masking(mask_value=0.0, name='effect_seq_mask')(effect_seq)

    emb = Concatenate(axis=-1, name='emb')([img_emb, effect_emb])

    x = Bidirectional(LSTM(lstm_dim))(emb)
    x = Dense(fc_dim, activation='elu')(x)
    x = Dropout(dropout)(x)
    out = Dense(n_effects, activation='softmax', name='next_effect')(x)

    model = Model([img_seq, effect_seq], out)
    return model


def next_effect_rnn_cnn(in_x: int = 128,
                        in_y: int = 88,
                        n_mfcc: int = 30,
                        n_channels: int = 2,
                        fc_dim: int = 128,
                        dropout: float = 0.5,
                        concat_input: bool = True) -> Tuple[Tensor, ...]:
    assert n_mfcc > 0
    assert n_channels == 2
    log.info(f'Using fc_dim of {fc_dim}')
    log.info(f'Using dropout of {dropout}')

    temp_concat = Input(shape=(in_x + n_mfcc, in_y, n_channels))

    if concat_input:
        mel_input = Lambda(lambda t: t[:, :in_x, :, :],
                           name='mel_input')(temp_concat)
    else:
        mel_input = Input(shape=(in_x, in_y, n_channels), name='mel_input')

    x = Conv2D(32,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(mel_input)
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
    mel_x = Flatten(name='mel_flatten')(x)

    if concat_input:
        mfcc_input = Lambda(lambda t: t[:, in_x:, :, :],
                            name='mfcc_input')(temp_concat)
    else:
        mfcc_input = Input(shape=(n_mfcc, in_y, n_channels), name='mfcc_input')

    x = BatchNormalization(axis=-1)(mfcc_input)
    x = Conv2D(32,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((2, 4))(x)
    x = Conv2D(64,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((2, 4))(x)
    x = Conv2D(64,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((2, 4))(x)
    mfcc_x = Flatten(name='mfcc_flatten')(x)

    x = Concatenate(axis=-1)([mel_x, mfcc_x])
    x = Dense(fc_dim, activation='elu', name='fc')(x)
    fc = Dropout(dropout)(x)

    if concat_input:
        return temp_concat, fc
    else:
        return mel_input, mfcc_input, fc


def next_effect_rnn(in_x: int = 128,
                    in_y: int = 88,
                    n_mfcc: int = 30,
                    n_channels: int = 2,
                    n_effects: int = 5,
                    lstm_dim: int = 128,
                    fc_dim: int = 128,
                    dropout: float = 0.5,
                    cnn_architecture: Callable = next_effect_rnn_cnn,
                    cnn_fc_dim: int = 128,
                    cnn_dropout: float = 0.5) -> Model:
    assert n_mfcc > 0
    assert n_channels == 2
    cnn_input, cnn_fc = cnn_architecture(in_x=in_x,
                                         in_y=in_y,
                                         n_mfcc=n_mfcc,
                                         fc_dim=cnn_fc_dim,
                                         n_channels=n_channels,
                                         dropout=cnn_dropout,
                                         concat_input=True)
    cnn = Model(cnn_input, cnn_fc, name='cnn')

    mel_seq = Input(shape=(None, in_x, in_y, n_channels), name='mel_seq')
    mfcc_seq = Input(shape=(None, n_mfcc, in_y, n_channels), name='mfcc_seq')

    temp_concat = Concatenate(axis=2)([mel_seq, mfcc_seq])

    masked = Masking(mask_value=0.0, name='img_seq_mask')(temp_concat)
    img_emb = TimeDistributed(cnn, name='img_emb')(masked)

    effect_seq = Input(shape=(None, n_effects + 1), name='effect_seq')
    effect_emb = Masking(mask_value=0.0, name='effect_seq_mask')(effect_seq)

    emb = Concatenate(axis=-1, name='emb')([img_emb, effect_emb])

    x = Bidirectional(LSTM(lstm_dim))(emb)
    x = Dense(fc_dim, activation='elu')(x)
    x = Dropout(dropout)(x)
    out = Dense(n_effects, activation='softmax', name='next_effect')(x)

    model = Model([mel_seq, mfcc_seq, effect_seq], out)
    return model


def next_effect_seq_only_rnn(in_x: int = 128,
                             in_y: int = 88,
                             n_mfcc: int = 30,
                             n_channels: int = 2,
                             n_effects: int = 5,
                             lstm_dim: int = 128,
                             fc_dim: int = 128,
                             dropout: float = 0.5,
                             cnn_architecture: Callable = next_effect_rnn_cnn,
                             cnn_fc_dim: int = 128,
                             cnn_dropout: float = 0.5) -> Model:
    assert n_mfcc > 0
    assert n_channels == 2
    mel_input, mfcc_input, cnn_fc = cnn_architecture(in_x=in_x,
                                                     in_y=in_y,
                                                     n_mfcc=n_mfcc,
                                                     fc_dim=cnn_fc_dim,
                                                     n_channels=n_channels,
                                                     dropout=cnn_dropout,
                                                     concat_input=False)

    effect_seq = Input(shape=(None, n_effects + 1), name='effect_seq')
    effect_emb = Masking(mask_value=0.0, name='effect_seq_mask')(effect_seq)
    lstm_emb = Bidirectional(LSTM(lstm_dim))(effect_emb)

    emb = Concatenate(axis=-1, name='emb')([cnn_fc, lstm_emb])

    x = Dense(fc_dim, activation='elu')(emb)
    x = Dropout(dropout)(x)
    out = Dense(n_effects, activation='softmax', name='next_effect')(x)

    model = Model([mel_input, mfcc_input, effect_seq], out)
    return model


def all_effects_cnn(in_x: int = 128,
                    in_y: int = 88,
                    n_mfcc: int = 30,
                    n_channels: int = 2,
                    n_effects: int = 5,
                    fc_dim: int = 128,
                    dropout: float = 0.5,
                    cnn_architecture: Callable = next_effect_rnn_cnn,
                    cnn_fc_dim: int = 128,
                    cnn_dropout: float = 0.5) -> Model:
    assert n_mfcc > 0
    assert n_channels == 2
    mel_input, mfcc_input, cnn_fc = cnn_architecture(in_x=in_x,
                                                     in_y=in_y,
                                                     n_mfcc=n_mfcc,
                                                     fc_dim=cnn_fc_dim,
                                                     n_channels=n_channels,
                                                     dropout=cnn_dropout,
                                                     concat_input=False)
    x = Dense(fc_dim, activation='elu')(cnn_fc)
    x = Dropout(dropout)(x)
    out = Dense(n_effects, activation='sigmoid', name='effects')(x)

    model = Model([mel_input, mfcc_input], out)
    return model


if __name__ == '__main__':
    # model = baseline_effect_rnn()
    # model = effect_rnn()
    # model = effect_seq_only_rnn()
    # model.summary()
    # model.save(os.path.join(MODELS_DIR, 'random_baseline_effect_rnn.h5'))
    exit()

    import numpy as np

    test_target_img = np.ones((3, 128, 88, 1))

    test_effect_seq = [
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
    ]

    test_img_seq = [
        [
            np.ones((128, 88, 1))
        ],
        [
            np.ones((128, 88, 1)),
            np.ones((128, 88, 1))
        ],
        [
            np.ones((128, 88, 1)),
            np.ones((128, 88, 1)),
            np.ones((128, 88, 1)),
            np.ones((128, 88, 1))
        ],
    ]

    new_seq = [[np.concatenate([img, target], axis=-1)
                for img in seq] for target, seq in zip(test_target_img, test_img_seq)]

    padded_effect_seq = pad_sequences(test_effect_seq,
                                      value=0.0,
                                      padding='post',
                                      dtype='float32')
    padded_img_seq = pad_sequences(new_seq,
                                   value=0.0,
                                   padding='post',
                                   dtype='float32')

    log.info(f'padded_img_seq shape = {padded_img_seq.shape}')
    log.info(f'padded_effect_seq shape = {padded_effect_seq.shape}')
    # herp = model.predict([padded_img_seq, padded_effect_seq])
    # derp = 1
