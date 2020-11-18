import shutil
import os
import random
import numpy as np
from tensorflow import random as tf_random
import yaml
from pathlib import Path
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import ipykernel    # needed when using many metrics, to avoid automatic verbose=2 output
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from image_utils import TensorBoardImage, ImagesAndMasksGenerator
import git
from gcp_utils import copy_folder_locally_if_missing, getSystemInfo, getLibVersions
from models import generate_compiled_segmentation_model
from metrics_utils import global_threshold

metadata_file_name = 'metadata.yaml'

tmp_directory = Path('./tmp')


def sample_image_and_mask_paths(generator, n_paths):
    rand_inds = [random.randint(0, len(generator.image_filenames) - 1) for _ in range(n_paths)]
    image_paths = list(np.asarray(generator.image_filenames)[rand_inds])
    mask_paths = [{c: list(np.asarray(generator.mask_filenames[c]))[i] for c in generator.mask_filenames} for i in rand_inds]
    return list(zip(image_paths, mask_paths))


def train(gcp_bucket, config_file, random_module_global_seed, numpy_random_global_seed, tf_random_global_seed, pretrained_model_id, message):

    # seed global random generators if specified; global random seeds here must be int or default None (no seed given)
    if random_module_global_seed is not None:
        random.seed(random_module_global_seed)
    if numpy_random_global_seed is not None:
        np.random.seed(numpy_random_global_seed)
    if tf_random_global_seed is not None:
        tf_random.set_seed(tf_random_global_seed)

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
        rescale=1. / 255,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        random_rotation=train_config['data_augmentation']['random_90-degree_rotations'],
        seed=None if 'training_data_shuffle_seed' not in train_config else train_config['training_data_shuffle_seed'])

    validation_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, train_config['dataset_id'],
             'validation').as_posix(),
        rescale=1. / 255,
        target_size=target_size,
        batch_size=batch_size,
        seed=None if 'validation_data_shuffle_seed' not in train_config else train_config['validation_data_shuffle_seed'])

    if pretrained_model_id is not None:
        local_pretrained_model_dir = Path(tmp_directory, 'pretrained_models')
        copy_folder_locally_if_missing(os.path.join(gcp_bucket, 'models', pretrained_model_id), local_pretrained_model_dir)
        path_pretrained_model = Path(local_pretrained_model_dir, pretrained_model_id, "model.hdf5").as_posix()

        with Path(local_pretrained_model_dir, pretrained_model_id, 'config.yaml').open('r') as f:
            pretrained_model_config = yaml.safe_load(f)['train_config']

        with Path(local_pretrained_model_dir, pretrained_model_id, 'metadata.yaml').open('r') as f:
            pretrained_model_metadata = yaml.safe_load(f)

        pretrained_info = {'pretrained_model_id': pretrained_model_id, 'pretrained_config': pretrained_model_config, 'pretrained_metadata': pretrained_model_metadata}

        # confirm that the current model and pretrained model configurations are compatible
        assert pretrained_model_config['segmentation_model']['model_name'] == train_config['segmentation_model']['model_name']
        assert pretrained_model_config['segmentation_model']['model_parameters']['backbone_name'] == train_config['segmentation_model']['model_parameters']['backbone_name']
        assert pretrained_model_config['segmentation_model']['model_parameters']['activation'] == train_config['segmentation_model']['model_parameters']['activation']
        assert pretrained_model_config['segmentation_model']['model_parameters']['input_shape'] == train_config['segmentation_model']['model_parameters']['input_shape']

        # same loss function
        assert pretrained_model_config['loss'] == train_config['loss']
        # confirm that the number of classes in pretrain is the same as train
        assert pretrained_model_metadata['num_classes'] == len(train_generator.mask_filenames)
        # same target size
        assert pretrained_model_metadata['dataset_config']['target_size'] == dataset_config['target_size']

    else:
        path_pretrained_model = None
        pretrained_info = None

    compiled_model = generate_compiled_segmentation_model(
        train_config['segmentation_model']['model_name'],
        train_config['segmentation_model']['model_parameters'],
        len(train_generator.mask_filenames),
        train_config['loss'],
        train_config['optimizer'],
        path_pretrained_model)

    model_checkpoint_callback = ModelCheckpoint(Path(model_dir, 'model.hdf5').as_posix(),
                                                monitor='loss', verbose=1, save_best_only=True)
    # profile_batch = 0 is needed until insufficinet privileges issue resolved with CUPTI
    #   (_https://github.com/tensorflow/tensorflow/issues/35860)
    tensorboard_callback = TensorBoard(log_dir=logs_dir.as_posix(), write_graph=True,
                                       write_grads=False, write_images=True, update_freq='epoch', profile_batch=0)

    n_sample_images = 20
    train_image_and_mask_paths = sample_image_and_mask_paths(train_generator, n_sample_images)
    validation_image_and_mask_paths = sample_image_and_mask_paths(validation_generator, n_sample_images)

    csv_logger_callback = CSVLogger(Path(model_dir, 'metrics.csv').as_posix(), append=True)

    results = compiled_model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[model_checkpoint_callback, tensorboard_callback, csv_logger_callback]
    )

    # individual plots
    metric_names = [m.name for m in compiled_model.metrics]
    for metric_name in metric_names:
        fig, ax = plt.subplots()
        for split in ['train', 'validate']:
            key_name = metric_name
            if split == 'validate':
                key_name = 'val_' + key_name
            ax.plot(range(epochs), results.history[key_name], label=split)
        ax.set_xlabel('epochs')
        if metric_name == 'loss' and hasattr(compiled_model.loss, '__name__'):
            ax.set_ylabel(compiled_model.loss.__name__)
        elif metric_name == 'loss' and hasattr(compiled_model.loss, 'name'):
            ax.set_ylabel(compiled_model.loss.name)
        else:
            ax.set_ylabel(metric_name)
        ax.legend()
        if metric_name == 'loss' and hasattr(compiled_model.loss, '__name__'):
            fig.savefig(Path(plots_dir, compiled_model.loss.__name__ + '.png').as_posix())
        elif metric_name == 'loss' and hasattr(compiled_model.loss, 'name'):
            fig.savefig(Path(plots_dir, compiled_model.loss.name + '.png').as_posix())
        else:
            fig.savefig(Path(plots_dir, metric_name + '.png').as_posix())
    plt.close()

    # mosaic of subplot
    if len(train_generator.mask_filenames) == 1:
        num_rows = 2
    else:  # 1 row for all classes, 1 row for each of n classes
        num_rows = len(train_generator.mask_filenames) + 1
    num_cols = np.ceil(len(metric_names) / num_rows).astype(int)
    fig2, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 3.25, num_rows * 3.25))
    counter_m = 0
    counter_n = 0
    for metric_name in metric_names:
        for split in ['train', 'validate']:
            key_name = metric_name
            if split == 'validate':
                key_name = 'val_' + key_name
            axes[counter_m, counter_n].plot(range(epochs), results.history[key_name], label=split)
        axes[counter_m, counter_n].set_xlabel('epochs')
        if metric_name == 'loss' and hasattr(compiled_model.loss, '__name__'):
            axes[counter_m, counter_n].set_ylabel(compiled_model.loss.__name__)
        elif metric_name == 'loss' and hasattr(compiled_model.loss, 'name'):
            axes[counter_m, counter_n].set_ylabel(compiled_model.loss.name)
        else:
            axes[counter_m, counter_n].set_ylabel(metric_name)
        axes[counter_m, counter_n].legend()
        counter_n += 1
        if counter_n == num_cols:  # plots per row
            counter_m += 1
            counter_n = 0
    fig2.tight_layout()
    fig2.savefig(Path(plots_dir, 'metrics_mosaic.png').as_posix())
    plt.close()

    metadata_sys = {
        'System_info': getSystemInfo(),
        'Lib_versions_info': getLibVersions()
    }

    metadata = {
        'message': message,
        'gcp_bucket': gcp_bucket,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'num_classes': len(train_generator.mask_filenames),
        'target_size': target_size,
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'original_config_filename': config_file,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1),
        'dataset_config': dataset_config,
        'global_threshold_for_metrics': global_threshold,
        'random-module-global-seed': random_module_global_seed,
        'numpy_random_global_seed': numpy_random_global_seed,
        'tf_random_global_seed': tf_random_global_seed,
        'pretrained_model_info': pretrained_info,
        'metadata_system': metadata_sys
    }

    with Path(model_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -r '{}' '{}'".format(Path(tmp_directory, 'models').as_posix(), gcp_bucket))

    print('\n Train/Val Metadata:')
    print(metadata)
    print('\n')

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
    argparser.add_argument(
        '--random-module-global-seed',
        type=int,
        default=None,
        help='The setting of random.seed(global seed), where global seed is int or default None (no seed given).')
    argparser.add_argument(
        '--numpy-random-global-seed',
        type=int,
        default=None,
        help='The setting of np.random.seed(global seed), where global seed is int or default None (no seed given).')
    argparser.add_argument(
        '--tf-random-global-seed',
        type=int,
        default=None,
        help='The setting of tf.random.set_seed(global seed), where global seed is int or default None (no seed given).')
    argparser.add_argument(
        '--pretrained-model-id',
        type=str,
        default=None,
        help='The model ID with previously trained weights.')
    argparser.add_argument(
        '--message',
        type=str,
        default=None,
        help='A str message the used wants to leave, the default is None.')
    train(**argparser.parse_args().__dict__)
