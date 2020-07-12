from typing import List, Dict, Any

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.python.framework.ops import Tensor


class TriosVAE(Model):
    def __init__(self,
                 enc: Model,
                 dec: Model,
                 is_conditional: bool,
                 **kwargs: Any) -> None:
        super(TriosVAE, self).__init__(**kwargs)
        self.enc = enc
        self.dec = dec
        self.is_conditional = is_conditional

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
