import shutil
import os
import random
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from image_utils import TensorBoardImage, ImagesAndMasksGenerator
import git
from gcp_utils import copy_folder_locally_if_missing
from models import generate_compiled_segmentation_model


metadata_file_name = 'metadata.yaml'
tmp_directory = Path('./tmp')


def sample_image_and_mask_paths(generator, n_paths):
    random.seed(0)
    rand_inds = [random.randint(0, len(generator.image_filenames)-1) for _ in range(n_paths)]
    image_paths = list(np.asarray(generator.image_filenames)[rand_inds])
    mask_paths = [{c: list(np.asarray(generator.mask_filenames[c]))[i] for c in generator.mask_filenames} for i in rand_inds]
    return list(zip(image_paths, mask_paths))


def train(gcp_bucket, config_file):

    start_dt = datetime.now()

    with Path(config_file).open('r') as f:
        train_config = yaml.safe_load(f)['train_config']

    assert "gs://" in gcp_bucket

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    local_dataset_dir = Path(tmp_directory, 'datasets')

    copy_folder_locally_if_missing(os.path.join(gcp_bucket, 'datasets', train_config['dataset_id']),
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
        yaml.safe_dump({'train_config': train_config}, f)

    target_size = dataset_config['target_size']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']

    train_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, train_config['dataset_id'], 'train').as_posix(),
        rescale=1./255,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        random_rotation=train_config['data_augmentation']['random_90-degree_rotations'],
        seed=train_config['training_data_shuffle_seed'])

    validation_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, train_config['dataset_id'],
             'validation').as_posix(),
        rescale=1./255,
        target_size=target_size,
        batch_size=batch_size)

    compiled_model = generate_compiled_segmentation_model(
        train_config['segmentation_model']['model_name'],
        train_config['segmentation_model']['model_parameters'],
        len(train_generator.mask_filenames),
        train_config['loss'],
        train_config['optimizer'])

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

    results = compiled_model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[model_checkpoint_callback, tensorboard_callback, tensorboard_image_callback, csv_logger_callback])

    metric_names = ['loss'] + [m.name for m in compiled_model.metrics]

    for metric_name in metric_names:

        fig, ax = plt.subplots()
        for split in ['train', 'validate']:

            key_name = metric_name
            if split == 'validate':
                key_name = 'val_' + key_name

            ax.plot(range(epochs), results.history[key_name], label=split)
        ax.set_xlabel('epochs')
        if metric_name == 'loss':
            ax.set_ylabel(compiled_model.loss.__name__)
        else:
            ax.set_ylabel(metric_name)
        ax.legend()
        if metric_name == 'loss':
            fig.savefig(Path(plots_dir, compiled_model.loss.__name__ + '.png').as_posix())
        else:
            fig.savefig(Path(plots_dir, metric_name + '.png').as_posix())

    metadata = {
        'gcp_bucket': gcp_bucket,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'num_classes': len(train_generator.mask_filenames),
        'target_size': target_size,
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'original_config_filename': config_file,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1),
        'dataset_config': dataset_config
    }

    with Path(model_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -r '{}' '{}'".format(Path(tmp_directory, 'models').as_posix(), gcp_bucket))

    shutil.rmtree(tmp_directory.as_posix())


if __name__ == "__main__":
    import argparse
    import sys

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument(
        '--gcp-bucket',
        type=str,
        help='The GCP bucket where the prepared data is located and to use to store the trained model.')
    argparser.add_argument(
        '--config-file',
        type=str,
        help='The location of the train configuration file.')

    train(**argparser.parse_args().__dict__)
