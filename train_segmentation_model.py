import shutil
import os
import random
import numpy as np
import yaml
from PIL import Image
from pathlib import Path
from datetime import datetime
import pytz
from keras.utils import Sequence
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from segmentation_models import Unet
from segmentation_models.metrics import iou_score
from image_utils import TensorBoardImage
from collections import OrderedDict
import git


metadata_file_name = 'metadata.yaml'
tmp_directory = Path('./tmp')


def sample_image_and_mask_paths(generator, n_paths):
    random.seed(0)
    rand_inds = [random.randint(0, len(generator.image_filenames)-1) for _ in range(n_paths)]
    image_paths = list(np.asarray(generator.image_filenames)[rand_inds])
    mask_paths = [{c: list(np.asarray(generator.mask_filenames[c]))[i] for c in generator.mask_filenames} for i in rand_inds]
    return list(zip(image_paths, mask_paths))


# adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class ImagesAndMasksGenerator(Sequence):
    def __init__(self, dataset_directory, rescale, target_size, batch_size, shuffle=True, seed=None, random_rotation=False):
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


def copy_dataset_locally_if_missing(dataset_remote_source, local_dataset_dir):
    if not os.path.exists(local_dataset_dir.as_posix()):
        local_dataset_dir.mkdir(parents=True, exist_ok=True)
        os.system("gsutil -m cp -r '{}' '{}'".format(dataset_remote_source, local_dataset_dir.as_posix()))


def main(config_file):

    with Path(config_file).open('r') as f:
        train_config = yaml.safe_load(f)['train_config']

    assert "gs://" in train_config['gcp_bucket']

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    local_dataset_dir = Path(tmp_directory, 'datasets')

    copy_dataset_locally_if_missing(os.path.join(train_config['gcp_bucket'], 'datasets', train_config['dataset_id']),
                                    local_dataset_dir)

    model_id = "{}_{}".format(train_config['model_id_prefix'], datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'))
    model_dir = Path(tmp_directory, 'models', model_id)
    model_dir.mkdir(parents=True)

    target_size = train_config['target_size']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']

    train_generator = ImagesAndMasksGenerator(Path(local_dataset_dir, train_config['dataset_id'], 'train').as_posix(),
                                              rescale=1./255,
                                              target_size=target_size,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              random_rotation=train_config['augmentation']['random_90-degree_rotations'],
                                              seed=None)

    validation_generator = ImagesAndMasksGenerator(Path(local_dataset_dir, train_config['dataset_id'],
                                                        'validation').as_posix(),
                                                   rescale=1./255,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   seed=None)

    model = Unet('vgg16', input_shape=(None, None, 1), classes=len(train_generator.mask_filenames), encoder_weights=None)

    loss_fn = 'binary_crossentropy' if len(train_generator.mask_filenames) == 1 else 'categorical_crossentropy'

    model.compile(optimizer=Adam(),
                  loss=loss_fn,
                  metrics=["accuracy", iou_score])

    logdir = Path(model_dir, 'logs')
    model_checkpoint_callback = ModelCheckpoint(Path(model_dir, 'model.hdf5').as_posix(),
                                                monitor='loss', verbose=1, save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=logdir.as_posix(), batch_size=batch_size, write_graph=True,
                                       write_grads=False, write_images=True, update_freq='epoch')

    n_sample_images = 20
    train_image_and_mask_paths = sample_image_and_mask_paths(train_generator, n_sample_images)
    validation_image_and_mask_paths = sample_image_and_mask_paths(validation_generator, n_sample_images)

    tensorboard_image_callback = TensorBoardImage(
        log_dir=logdir.as_posix(),
        images_and_masks_paths=train_image_and_mask_paths + validation_image_and_mask_paths)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[model_checkpoint_callback, tensorboard_callback, tensorboard_image_callback])

    metadata = {
        'gcp_bucket': train_config['gcp_bucket'],
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'config_file': {
            'file_name': config_file,
            'contents': train_config
        }
    }

    with Path(model_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -r '{}' '{}'".format(Path(tmp_directory, 'models').as_posix(),
                                                 os.path.join(train_config['gcp_bucket'], 'models')))

    shutil.rmtree(tmp_directory.as_posix())


if __name__ == "__main__":
    import argparse
    import sys

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument(
        '--config-file',
        type=str,
        help='The location of the train configuration file.')

    main(**argparser.parse_args().__dict__)
