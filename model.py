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


class SmallBasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, name=None):
        super().__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            name='conv1'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=int(filters / 4),
            kernel_size=(3, 1),
            strides=(1, 1),
            padding='same',
            name='conv2'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=int(filters / 4),
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            name='conv3'
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            name='conv4'
        )

        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return tf.keras.layers.ReLU()(x)


class LPRNet(tf.keras.Model):
    def __init__(self, num_chars, name=None):
        super().__init__(name=name)

        self.input_layer = tf.keras.layers.Input(shape=(24, 94, 3))
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='conv1'
        )
        self.maxpool1 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='maxpool1'
        )
        self.bb1 = SmallBasicBlock(filters=64, name='bb1')
        self.maxpool2 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='maxpool2'
        )
        self.bb2 = SmallBasicBlock(filters=128, name='bb2')
        self.bb3 = SmallBasicBlock(filters=128, name='bb3')
        self.maxpool3 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='maxpool3'
        )
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5, name='dropout1')
        self.conv2 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(4, 1),
            strides=(1, 1),
            padding='valid',
            name='conv2'
        )
        self.dropout2 = tf.keras.layers.Dropout(rate=0.5, name='dropout2')
        self.conv3 = tf.keras.layers.Conv2D(
            filters=num_chars,
            kernel_size=(1, 13),
            strides=(1, 1),
            padding='same',
            name='conv3'
        )

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.conv1(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.maxpool1(x)
        x = self.bb1(x)
        x = self.maxpool2(x)
        x = self.bb2(x)
        x = self.bb3(x)
        x = self.maxpool3(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        return x
