import shutil
import os
import random
import numpy as np
from tensorflow import random as tf_random
import yaml
from pathlib import Path
from datetime import datetime
import pytz
import git
from gcp_utils import copy_folder_locally_if_missing
from image_utils import ImagesAndMasksGenerator
from models import generate_compiled_segmentation_model
from metrics_utils import global_threshold

# test can be run multiple times (with or without optimized thresholds, global thresholds), create new each time
test_datetime = datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ')
metadata_file_name = 'metadata_' + test_datetime + '.yaml'
tmp_directory = Path('./tmp')


def test(gcp_bucket, dataset_id, model_id, batch_size, trained_thresholds_id, random_module_global_seed,
         numpy_random_global_seed, tf_random_global_seed):

    # seed global random generators if specified; global random seeds here must be int or default None (no seed given)
    if random_module_global_seed is not None:
        random.seed(random_module_global_seed)
    if numpy_random_global_seed is not None:
        np.random.seed(numpy_random_global_seed)
    if tf_random_global_seed is not None:
        tf_random.set_seed(tf_random_global_seed)

    start_dt = datetime.now()

    assert "gs://" in gcp_bucket

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    local_dataset_dir = Path(tmp_directory, 'datasets')
    local_model_dir = Path(tmp_directory, 'models')

    copy_folder_locally_if_missing(os.path.join(gcp_bucket, 'datasets', dataset_id), local_dataset_dir)

    copy_folder_locally_if_missing(os.path.join(gcp_bucket, 'models', model_id), local_model_dir)

    test_id = "{}_{}".format(model_id, dataset_id)
    test_dir = Path(tmp_directory, 'tests', test_id)
    test_dir.mkdir(parents=True)

    with Path(local_dataset_dir, dataset_id, 'config.yaml').open('r') as f:
        dataset_config = yaml.safe_load(f)['dataset_config']

    with Path(local_model_dir, model_id, 'config.yaml').open('r') as f:
        train_config = yaml.safe_load(f)['train_config']

    if trained_thresholds_id is not None:
        with Path(local_model_dir, model_id, trained_thresholds_id).open('r') as f:
            threshold_output_data = yaml.safe_load(f)

    target_size = dataset_config['target_size']

    test_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, dataset_id, 'test').as_posix(),
        rescale=1./255,
        target_size=target_size,
        batch_size=batch_size,
        seed=None if 'test_data_shuffle_seed' not in train_config else train_config['test_data_shuffle_seed'])

    optimized_class_thresholds = {}
    if trained_thresholds_id is not None and 'thresholds_training_output' in threshold_output_data['metadata']:
        for i in range(len(test_generator.mask_filenames)):
            if ('x' in threshold_output_data['metadata']['thresholds_training_output'][str('class' + str(i))] and
                    threshold_output_data['metadata']['thresholds_training_output'][str('class' + str(i))]['success']):
                optimized_class_thresholds.update(
                    {str('class' + str(i)): threshold_output_data['metadata']['thresholds_training_output'][str('class' + str(i))]['x']}
                )
            else:
                AssertionError('Unsuccessfully trained threshold attempted to be loaded.')
    else:
        optimized_class_thresholds = None

    compiled_model = generate_compiled_segmentation_model(
        train_config['segmentation_model']['model_name'],
        train_config['segmentation_model']['model_parameters'],
        len(test_generator.mask_filenames),
        train_config['loss'],
        train_config['optimizer'],
        Path(local_model_dir, model_id, "model.hdf5").as_posix(),
        optimized_class_thresholds=optimized_class_thresholds)

    results = compiled_model.evaluate(test_generator)

    if hasattr(compiled_model.loss, '__name__'):
        metric_names = [compiled_model.loss.__name__] + [m.name for m in compiled_model.metrics]
    elif hasattr(compiled_model.loss, 'name'):
        metric_names = [compiled_model.loss.name] + [m.name for m in compiled_model.metrics]

    with Path(test_dir, str('metrics_' + test_datetime + '.csv')).open('w') as f:
        f.write(','.join(metric_names) + '\n')
        f.write(','.join(map(str, results)))

    metadata = {
        'gcp_bucket': gcp_bucket,
        'dataset_id': dataset_id,
        'model_id': model_id,
        'trained_thresholds_id': trained_thresholds_id,
        'trained_class_thresholds_loaded': optimized_class_thresholds,  # global thresh used if None
        'default_global_threshold_for_reference': global_threshold,
        'batch_size': batch_size,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1),
        'dataset_config': dataset_config,
        'train_config': train_config,
        'random-module-global-seed': random_module_global_seed,
        'numpy_random_global_seed': numpy_random_global_seed,
        'tf_random_global_seed': tf_random_global_seed
    }

    with Path(test_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -n -r '{}' '{}'".format(Path(tmp_directory, 'tests').as_posix(), gcp_bucket))

    print('\n Test Metadata:')
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
        help='The GCP bucket where the raw data is located and to use to store the processed stacks.')
    argparser.add_argument(
        '--dataset-id',
        type=str,
        help='The dataset ID.')
    argparser.add_argument(
        '--model-id',
        type=str,
        help='The model ID.')
    argparser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='The batch size to use during inference.')
    argparser.add_argument(
        '--trained-thresholds-id',
        type=str,
        default=None,
        help='The specified trained thresholds file id.')
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

    test(**argparser.parse_args().__dict__)
