#      CVLPR
#
#      Copyright (c) 2023. Risang Baskoro
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see https://www.gnu.org/licenses/.

import tensorflow as tf

from model import SmallBasicBlock

if __name__ == '__main__':
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(24, 94, 3)),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='conv1'
        ),
        tf.keras.layers.BatchNormalization(name='bn1'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='maxpool1'
        ),
        SmallBasicBlock(filters=64, name='bb1'),
        tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='maxpool2'
        ),
        SmallBasicBlock(filters=64, name='bb2'),
        SmallBasicBlock(filters=64, name='bb3'),
        tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='maxpool3'
        ),
        tf.keras.layers.Dropout(rate=0.5, name='dropout1'),
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(4, 1),
            strides=(1, 1),
            padding='valid',
            name='conv2'
        ),
        tf.keras.layers.BatchNormalization(name='bn2'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(rate=0.5, name='dropout2'),
        tf.keras.layers.Conv2D(
            filters=len(CHARS),
            kernel_size=(1, 13),
            strides=(1, 1),
            padding='same',
            name='conv3'
        ),
        tf.keras.layers.BatchNormalization(name='bn3'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Lambda(tf.reduce_mean, arguments={'axis': 1}, name='global_avg_pool'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()
