import shutil
import os
import random
import numpy as np
import yaml
from PIL import Image
from pathlib import Path
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from segmentation_models import Unet
from segmentation_models.metrics import iou_score
from image_utils import TensorBoardImage, ImagesAndMasksGenerator
import git
from gcp_utils import copy_dataset_locally_if_missing


metadata_file_name = 'metadata.yaml'
tmp_directory = Path('./tmp')


def sample_image_and_mask_paths(generator, n_paths):
    random.seed(0)
    rand_inds = [random.randint(0, len(generator.image_filenames)-1) for _ in range(n_paths)]
    image_paths = list(np.asarray(generator.image_filenames)[rand_inds])
    mask_paths = [{c: list(np.asarray(generator.mask_filenames[c]))[i] for c in generator.mask_filenames} for i in rand_inds]
    return list(zip(image_paths, mask_paths))


def train(config_file):

    start_dt = datetime.now()

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

    plots_dir = Path(model_dir, 'plots')
    plots_dir.mkdir(parents=True)

    logs_dir = Path(model_dir, 'logs')
    logs_dir.mkdir(parents=True)

    with Path(local_dataset_dir, train_config['dataset_id'], 'config.yaml').open('r') as f:
        dataset_config = yaml.safe_load(f)['dataset_config']

    with Path(model_dir, 'config.yaml').open('w') as f:
        yaml.safe_dump({'train_config': dataset_config}, f)

    target_size = dataset_config['target_size']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']

    train_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, train_config['dataset_id'], 'train').as_posix(),
        rescale=1./255,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        random_rotation=train_config['augmentation']['random_90-degree_rotations'],
        seed=None)

    validation_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, train_config['dataset_id'],
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

    model_checkpoint_callback = ModelCheckpoint(Path(model_dir, 'model.hdf5').as_posix(),
                                                monitor='loss', verbose=1, save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=logs_dir.as_posix(), batch_size=batch_size, write_graph=True,
                                       write_grads=False, write_images=True, update_freq='epoch')

    n_sample_images = 20
    train_image_and_mask_paths = sample_image_and_mask_paths(train_generator, n_sample_images)
    validation_image_and_mask_paths = sample_image_and_mask_paths(validation_generator, n_sample_images)

    tensorboard_image_callback = TensorBoardImage(
        log_dir=logs_dir.as_posix(),
        images_and_masks_paths=train_image_and_mask_paths + validation_image_and_mask_paths)

    csv_logger_callback = CSVLogger(Path(model_dir, 'metrics.csv').as_posix(), append=True)

    results = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[model_checkpoint_callback, tensorboard_callback, tensorboard_image_callback, csv_logger_callback])

    metric_names = ['loss', 'accuracy', 'iou_score']
    for metric_name in metric_names:

        fig, ax = plt.subplots()
        for split in ['train', 'validate']:

            key_name = metric_name
            if split == 'validate':
                key_name = 'val_' + key_name

            ax.plot(range(epochs), results.history[key_name], label=split)
        ax.set_xlabel('episodes')
        if metric_name == 'loss':
            ax.set_ylabel(loss_fn)
        else:
            ax.set_ylabel(metric_name)
        ax.legend()
        if metric_name == 'loss':
            fig.savefig(Path(plots_dir, loss_fn + '.png').as_posix())
        else:
            fig.savefig(Path(plots_dir, metric_name + '.png').as_posix())

    metadata = {
        'gcp_bucket': train_config['gcp_bucket'],
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'original_config_filename': config_file,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1)
    }

    with Path(model_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -r '{}' '{}'".format(Path(tmp_directory, 'models').as_posix(),
                                                 train_config['gcp_bucket']))

    shutil.rmtree(tmp_directory.as_posix())


if __name__ == "__main__":
    import argparse
    import sys

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument(
        '--config-file',
        type=str,
        help='The location of the train configuration file.')

    train(**argparser.parse_args().__dict__)
