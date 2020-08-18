import os
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, \
    Conv2D, Dropout, Bidirectional, Permute, Lambda, Conv1D, LSTM, Concatenate
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.utils.np_utils import to_categorical

from python.config import DATASETS_DIR


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


if __name__ == '__main__':
    data_path = os.path.join(DATASETS_DIR, 'mels_10k_testing_midi.npz')
    data = np.load(data_path)
    mels = data['mels']
    params = data['params']
    print(mels.shape)
    print(mels.dtype)
    print(params.shape)
    print(params.dtype)
    x = np.expand_dims(mels, axis=-1)
    x_midi = params[:, 4:5]
    x_midi = to_categorical(x_midi, num_classes=86)
    y_reg = params[:, 1:2].astype(np.float32) / 100
    y_class = (params[:, 1:2].astype(np.float32) / 100) * 13
    y_class += 0.5
    y_class = y_class.astype(np.int32)
    print(x.shape)
    print(x.dtype)
    print(x_midi.shape)
    print(x_midi.dtype)
    print(y_reg.shape)
    print(y_reg.dtype)
    print(y_class.shape)
    print(y_class.dtype)

    cnn = build_cnn3_classifier(mel_spec_x=128,
                                mel_spec_y=88,
                                n_reg=1,
                                n_class=14)
    cnn.summary()
    cnn.compile(optimizer='adam', loss=['mse', 'sparse_categorical_crossentropy'], metrics=['mae', 'acc'])

    cnn.fit([x, x_midi], [y_reg, y_class], batch_size=64, epochs=100, validation_split=0.2, shuffle=True)
