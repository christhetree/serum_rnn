import logging
import os
from typing import List, Dict, Any, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, \
    Conv2D, Dropout, Bidirectional, Permute, Lambda, Conv1D, LSTM, Concatenate
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.python.framework.ops import Tensor

from config import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

GPU = 0
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    log.info(f'GPUs available: {physical_devices}')
    tf.config.experimental.set_visible_devices(physical_devices[GPU], 'GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[GPU], enable=True)


def build_baseline_cnn(
        mel_spec_x: int = 128,
        mel_spec_y: int = 131,
        n_bin: int = 0,
        n_cate: List[int] = [],
        cate_names: List[str] = [],
        n_cont: int = 0,
        fc_dim: int = 128,
        dropout_rate: float = 0.50) -> Model:
    input_img = Input(shape=(mel_spec_x, mel_spec_y, 1))
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
    x = Dropout(dropout_rate)(x)

    outputs = []
    if n_bin:
        bin_output = Dense(n_bin, activation='sigmoid', name='bin_output')(x)
        outputs.append(bin_output)

    for n_classes, name in zip(n_cate, cate_names):
        cate_output = Dense(n_classes, activation='softmax', name=name)(x)
        outputs.append(cate_output)

    if n_cont:
        cont_output = Dense(n_cont, activation='linear', name='cont_output')(x)
        outputs.append(cont_output)

    assert outputs
    model = Model(input_img, outputs)
    return model


def build_cnn3_classifier(
        mel_spec_x: int = 128,
        mel_spec_y: int = 131,
        n_reg: int = 1,
        n_class: int = 14,
        n_midi: int = 86,
        dropout_rate: float = 0.50) -> Model:
    input_midi = Input(shape=(n_midi,))
    input_img = Input(shape=(mel_spec_x, mel_spec_y, 1))
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
    x = Concatenate()([x, input_midi])
    x = Dense(128, activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    reg_output = Dense(n_reg, activation='linear')(x)
    class_output = Dense(n_class, activation='softmax')(x)
    model = Model([input_img, input_midi], [reg_output, class_output])
    return model


def build_lstm_classifier(
        mel_spec_x: int = 128,
        mel_ts: int = 128,
        n_class: int = 9,
        dropout_rate: float = 0.50,
        lstm_dim: int = 128,
        fc_dim: int = 128) -> Model:
    input_img = Input(shape=(mel_spec_x, mel_ts, 1))
    x = Lambda(lambda t: K.squeeze(t, axis=-1))(input_img)
    x = Permute((2, 1))(x)
    x = Bidirectional(LSTM(lstm_dim))(x)
    x = Dense(fc_dim, activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(n_class, activation='softmax')(x)
    model = Model(input_img, x)
    return model


def build_cnn_lstm_classifier(
        mel_spec_x: int = 128,
        mel_ts: int = 128,
        n_class: int = 9,
        dropout_rate: float = 0.50,
        lstm_dim: int = 128,
        fc_dim: int = 128) -> Model:
    input_img = Input(shape=(mel_spec_x, mel_ts, 1))
    x = Lambda(lambda t: K.squeeze(t, axis=-1))(input_img)
    x = Permute((2, 1))(x)
    x = Conv1D(64,
               3,
               strides=1,
               padding='same',
               activation='elu')(x)
    x = Bidirectional(LSTM(lstm_dim))(x)
    x = Dense(fc_dim, activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(n_class, activation='softmax')(x)
    model = Model(input_img, x)
    return model


class MLPBaseline(Model):
    def __init__(self,
                 **kwargs: Any) -> None:
        super(MLPBaseline, self).__init__(**kwargs)

    def call(self, inputs: List[Tensor]) -> List[Tensor]:
        z_mean, z_log_var, z = self.enc(inputs)

        if self.is_conditional:
            cond = inputs[-1]
            return self.dec([z, cond])
        else:
            return self.dec(z)

    def rec_loss(self, data: Tensor, reconstruction: Tensor) -> Tensor:
        rec_loss = tf.reduce_mean(
            sparse_categorical_crossentropy(data, reconstruction)
        )
        # rec_loss *= 32.0
        return rec_loss

    def rec_acc(self, data: Tensor, reconstruction: Tensor) -> Tensor:
        rec_acc = tf.reduce_mean(
            sparse_categorical_accuracy(data, reconstruction)
        )
        return rec_acc

    def train_step(self,
                   data: (List[Tensor], List[Tensor])) -> Dict[str, float]:
        if isinstance(data, tuple):
            data = data[0]  # Only care about X since it's an autoencoder
        with tf.GradientTape() as tape:
            if self.is_conditional:
                melody_data, bass_data, drums_data, cond_data = data
                z_mean, z_log_var, z = self.enc(data)
                melody_rec, bass_rec, drums_rec = self.dec([z, cond_data])
            else:
                melody_data, bass_data, drums_data = data
                z_mean, z_log_var, z = self.enc(data)
                melody_rec, bass_rec, drums_rec = self.dec(z)

            melody_rec_loss = self.rec_loss(melody_data, melody_rec)
            bass_rec_loss = self.rec_loss(bass_data, bass_rec)
            drums_rec_loss = self.rec_loss(drums_data, drums_rec)
            rec_loss = melody_rec_loss + bass_rec_loss + drums_rec_loss

            melody_rec_acc = self.rec_acc(melody_data, melody_rec)
            bass_rec_acc = self.rec_acc(bass_data, bass_rec)
            drums_rec_acc = self.rec_acc(drums_data, drums_rec)
            rec_acc = tf.reduce_mean([melody_rec_acc, bass_rec_acc, drums_rec_acc])

            kl_loss = 1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5

            total_loss = rec_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            'loss': total_loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
            'm_rec_loss': melody_rec_loss,
            'b_rec_loss': bass_rec_loss,
            'd_rec_loss': drums_rec_loss,
            'rec_acc': rec_acc,
            'm_rec_acc': melody_rec_acc,
            'b_rec_acc': bass_rec_acc,
            'd_rec_acc': drums_rec_acc,
        }

    def test_step(self,
                  data: (List[Tensor], List[Tensor])) -> Dict[str, float]:
        if isinstance(data, tuple):
            data = data[0]  # Only care about X since it's an autoencoder

        if self.is_conditional:
            melody_data, bass_data, drums_data, cond_data = data
            z_mean, z_log_var, z = self.enc(data)
            melody_rec, bass_rec, drums_rec = self.dec([z, cond_data])
        else:
            melody_data, bass_data, drums_data = data
            z_mean, z_log_var, z = self.enc(data)
            melody_rec, bass_rec, drums_rec = self.dec(z)

        melody_rec_loss = self.rec_loss(melody_data, melody_rec)
        bass_rec_loss = self.rec_loss(bass_data, bass_rec)
        drums_rec_loss = self.rec_loss(drums_data, drums_rec)
        rec_loss = melody_rec_loss + bass_rec_loss + drums_rec_loss

        melody_rec_acc = self.rec_acc(melody_data, melody_rec)
        bass_rec_acc = self.rec_acc(bass_data, bass_rec)
        drums_rec_acc = self.rec_acc(drums_data, drums_rec)
        rec_acc = tf.reduce_mean([melody_rec_acc, bass_rec_acc, drums_rec_acc])

        kl_loss = 1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5

        total_loss = rec_loss + kl_loss

        return {
            'loss': total_loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
            'm_rec_loss': melody_rec_loss,
            'b_rec_loss': bass_rec_loss,
            'd_rec_loss': drums_rec_loss,
            'rec_acc': rec_acc,
            'm_rec_acc': melody_rec_acc,
            'b_rec_acc': bass_rec_acc,
            'd_rec_acc': drums_rec_acc,
        }


def train_rhythm_vae(
        vae: Model,
        x: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
        model_name: str,
        batch_size: int = 64,
        epochs: int = 100,
        val_split: float = 0.05,
        patience: int = 8,
        output_dir_path: str = '../../out/model_checkpoints/') -> None:
    save_path = os.path.join(
        output_dir_path, model_name + '_e{epoch:02d}_vl{val_loss:.4f}.h5')
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=patience,
                       verbose=1,
                       mode='auto')
    cp = ModelCheckpoint(
        save_path, monitor='val_loss', verbose=1, save_best_only=True)
    vae.fit(x,
            y,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=val_split,
            callbacks=[es, cp],
            verbose=1)


if __name__ == '__main__':
    # x_data_path = 'audio_render_test/default__sr_44100__nl_1.00__rl_1.00__vel_127__midi_040/distortion__gran_1000/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_14014.npz'
    x_data_path = 'audio_render_test/default__sr_44100__nl_1.00__rl_1.00__vel_127__midi_040/flanger__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_256__hop_len_256__norm_audio_F__norm_mel_T__n_14000.npz'
    x_data = np.load(os.path.join(DATA_DIR, x_data_path))
    x = x_data['mels']
    log.info(f'mels shape = {x.shape}')

    # y_data_path = f'{os.path.splitext(x_data_path)[0]}__y_97_99.npz'
    y_data_path = f'{os.path.splitext(x_data_path)[0]}__y_105_106_107_108.npz'

    n_bin = 0
    n_cate = []
    cate_names = []
    n_cont = 0
    y_s = []
    y_losses = {}
    metrics = {}

    y_data = np.load(os.path.join(DATA_DIR, y_data_path))

    if 'binary_params' in y_data:
        bin_params = y_data['binary_params'].tolist()
        n_bin = len(bin_params)
        y_bin = y_data['binary']
        log.info(f'y_bin shape = {y_bin.shape}')
        assert y_bin.shape[-1] == n_bin
        y_s.append(y_bin)
        y_losses['bin_output'] = 'bce'
        metrics['bin_output'] = 'acc'

    if 'categorical_params' in y_data:
        cate_params = y_data['categorical_params'].tolist()
        n_cate = y_data['n_categories'].tolist()
        cate_names = y_data['param_to_desc'].tolist()

        for n_classes, desc in zip(n_cate, cate_names):
            y_cate = y_data[desc]
            log.info(f'{desc} n_classes = {n_classes}')
            log.info(f'{desc} max = {np.max(y_cate)}')
            log.info(f'{desc} min = {np.min(y_cate)}')
            assert 0 <= np.max(y_cate) < n_classes
            assert 0 <= np.min(y_cate) < n_classes
            y_s.append(y_cate)
            y_losses[desc] = 'sparse_categorical_crossentropy'
            metrics[desc] = 'acc'

    if 'continuous_params' in y_data:
        cont_params = y_data['continuous_params'].tolist()
        n_cont = len(cont_params)
        y_cont = y_data['continuous']
        assert y_cont.shape[-1] == n_cont
        log.info(f'y_cont shape = {y_cont.shape}')
        y_s.append(y_cont)
        y_losses['cont_output'] = 'mse'
        metrics['cont_output'] = 'mae'

    log.info(f'y_losses = {y_losses}')
    log.info(f'metrics = {metrics}')

    cnn = build_baseline_cnn(mel_spec_x=256,
                             mel_spec_y=175,
                             n_bin=n_bin,
                             n_cate=n_cate,
                             cate_names=cate_names,
                             n_cont=n_cont)

    cnn.summary()
    cnn.compile(optimizer='adam',
                loss=y_losses,
                metrics=metrics)

    cnn.fit(x,
            y_s,
            batch_size=64,
            epochs=64,
            validation_split=0.2,
            shuffle=True)
