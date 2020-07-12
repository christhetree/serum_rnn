import os
from typing import Union, List

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_vae(
        vae: Model,
        x: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
        model_name: str,
        batch_size: int = 512,
        epochs: int = 100,
        val_split: float = 0.10,
        patience: int = 8,
        output_dir_path: str = '../out/') -> None:
    save_path = os.path.join(
        output_dir_path,
        # model_name + '_e{epoch:02d}_vl{val_loss:.4f}_vkl{val_kl_loss:.4f}.h5'
        model_name + '_e{epoch:02d}_vl{val_loss:.4f}.h5'
    )
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
            verbose=1,
            validation_split=val_split,
            callbacks=[es, cp])
