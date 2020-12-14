# -*- coding : utf-8-*-
import os
import sys

import numpy
import matplotlib.image as mpimg  # read image
from keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):

    def __init__(self, files, batch_size=1, shuffle=True):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.files = files
        self.indexes = numpy.arange(len(self.files))
        self.shuffle = shuffle

    def __len__(self):
        """return: steps num of one epoch. """
        return len(self.files) // self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """

        # get batch data inds.
        batch_inds = self.indexes[index *
                                  self.batch_size:(index + 1) * self.batch_size]
        # get batch data file name.
        batch_files = [self.files[k] for k in batch_inds]

        # read batch data
        batch_images, batch_labels = self._read_data(batch_files)

        return batch_images, batch_labels

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            numpy.random.shuffle(self.indexes)

    def _read_data(self, batch_files):
        """Read a batch data.
        ---
        # Arguments
            batch_files: the file of batch data.

        # Returns
            images: (batch_size, (image_shape)).
            labels: (batch_size, (label_shape)). """

        images = []
        labels = []

        for file in batch_files:
            image = mpimg.imread('data/Images/' + file + '.jpg')
            images.append(image)
            lable = numpy.loadtxt('data/labels/' + file + '.arr', dtype=int)
            labels.append(lable)  # to one hot

        return numpy.array(images), numpy.array(labels)


if __name__ == '__main__':
    base_dir = os.path.join(sys.path[1], '')
    images_files = base_dir + 'data/cat_dog_image/'
    allfiles = []
    for file in images_files:
        allfiles.append(file.split('.')[0])  # get image name
    N = len(allfiles)
    train_files = allfiles[:int(N * 0.8)]
    test_files = allfiles[int(N * 0.8):]
    generator = DataGenerator(files=train_files, batch_size=32, shuffle=True)
    [a,b] = generator.__getitem__(1)