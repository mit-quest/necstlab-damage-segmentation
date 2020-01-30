import random
from collections import OrderedDict
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import keras as keras
from PIL import Image
import io


class_RGB_mapping = {
    'class_0': None,
    'class_1': [255, 0, 0],  # blue: 0 degree ply damage
    'class_2': [0, 255, 0],  # green: 45 degree ply damage
    'class_3': [0, 0, 255],  # red: 90 degree ply damage
}

GV_RGB_mapping = {
    100: [255, 0, 0],  # blue: 0 degree ply damage
    175: [0, 255, 0],  # green: 45 degree ply damage
    250: [0, 0, 255],  # red: 90 degree ply damage
}


def overlay_masks(images, masks):
    # assert type(images) == type(masks)
    if type(images) == np.ndarray:
        raise NotImplementedError
    else:  # assume is a file path
        image_file, mask_files = images, masks
        image = np.asarray(Image.open(image_file).convert('RGB'))
        masks = {c: np.asarray(Image.open(mask_files[c])) for c in mask_files}

        composite_image = image.copy()
        for c in masks:
            if class_RGB_mapping[c] is not None:
                composite_image[masks[c].astype(bool)] = class_RGB_mapping[c]

    composite_image = Image.fromarray(composite_image)
    return composite_image


def overlay_annotation(image_file, annotation_file):
    image = np.asarray(Image.open(image_file).convert('RGB'))
    annotation = np.asarray(Image.open(annotation_file))

    composite_image = image.copy()
    for annotation_gv in GV_RGB_mapping:
        composite_image[annotation == annotation_gv] = GV_RGB_mapping[annotation_gv]

    composite_image = Image.fromarray(composite_image)
    return composite_image


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, log_dir, images_and_masks_paths):
        super().__init__()
        self.log_dir = log_dir
        self.images_and_masks_paths = images_and_masks_paths

    def on_epoch_end(self, epoch, logs=None):

        writer = tf.summary.FileWriter(self.log_dir)

        for image_file, mask_files in self.images_and_masks_paths:
            composite_image = np.asarray(overlay_masks(image_file, mask_files))
            summary_image = make_image(composite_image)

            summary = tf.Summary(value=[tf.Summary.Value(tag=image_file.as_posix(), image=summary_image)])
            writer.add_summary(summary, epoch)

        writer.close()


# adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class ImagesAndMasksGenerator(Sequence):
    def __init__(self, dataset_directory, rescale, target_size, batch_size, shuffle=False, seed=None, random_rotation=False):
        self.dataset_directory = dataset_directory
        self.image_filenames = sorted(Path(self.dataset_directory, 'images').iterdir())
        self.mask_filenames = OrderedDict()
        for c in sorted(Path(self.dataset_directory, 'masks').iterdir()):
            self.mask_filenames[c.name] = sorted(c.iterdir())
        self.rescale = rescale
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.random_rotation = random_rotation
        self.indexes = None
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_image_filenames = [self.image_filenames[k] for k in indexes]
        batch_mask_filenames = {}
        for c in self.mask_filenames:
            batch_mask_filenames[c] = [self.mask_filenames[c][k] for k in indexes]

        # Generate data
        images, masks = self.__data_generation(batch_image_filenames, batch_mask_filenames)

        return images, masks

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_filenames, batch_mask_filenames):
        images = np.empty((self.batch_size, *self.target_size, 1))
        masks = np.empty((self.batch_size, *self.target_size, len(self.mask_filenames)), dtype=int)

        for i in range(len(batch_image_filenames)):
            rotation = 0
            if self.random_rotation:
                rotation = random.sample([0, 90, 180, 270], k=1)[0]
            images[i, :, :, 0] = np.asarray(Image.open(batch_image_filenames[i]).rotate(rotation))
            for j, c in enumerate(self.mask_filenames):
                masks[i, :, :, j] = np.asarray(Image.open(batch_mask_filenames[c][i]).rotate(rotation))

        images = images * self.rescale

        return images, masks
