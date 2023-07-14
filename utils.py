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

import os

import numpy as np

from const import CHARS_DICT


class TextDictionaryGenerator:
    """This class will generate the labels and training batch from the given data directory.
    Arguments:
        data_dir: The directory of the images. The images filename should be in the format of 'label_*.ext'
            where 'label' is the license plate number and '.ext' is the extension of the file.
        chars_dict: The dictionary of the characters. The key is the character and the value is the index of the
            character.
        labels_max_len: The maximum length of the labels. The labels will be padded with zeros if the length is less
            than the maximum length.
    """

    def __init__(self, data_dir, chars_dict, labels_max_len=9):
        self.filenames = []
        self.labels = []

        self._data_dir = data_dir
        self._chars_dict = chars_dict
        self._labels_max_len = labels_max_len

        self._num_examples = 0
        self._next_index = 0
        self._num_epochs = 0

        fs = os.listdir(self._data_dir)
        for filename in fs:
            self.filenames.append(filename)
            label = filename.split('_')[0]

            label = self._encode_label(label)
            self.labels.append(label)
            self._num_examples += 1

        self.labels = np.array(self.labels)

    def _encode_label(self, chars):
        """ Encode the label from the given filenames to NumPy array.
        Arguments:
            chars: The label from the given filenames.
        """
        label = list(np.zeros([self._labels_max_len]))
        for i, char in enumerate(chars):
            label[i] = self._chars_dict[char]
        return label


if __name__ == '__main__':
    gen = TextDictionaryGenerator('train', CHARS_DICT)
    print(gen.labels.shape)
