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
from gcp_utils import copy_folder_locally_if_missing, copy_file_locally_if_missing, getSystemInfo, getLibVersions
from image_utils import ImagesAndMasksGenerator
from models import train_prediction_thresholds, thresholds_training_history

# can train same model repeatedly with different optimization configurations
output_file_name = 'model_thresholds_' + datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ') + '.yaml'
tmp_directory = Path('./tmp')


def train_segmentation_model_prediction_thresholds(gcp_bucket, dataset_directory, model_id, batch_size,
                                                   optimizing_class_metric, dataset_downsample_factor,
                                                   random_module_global_seed, numpy_random_global_seed,
                                                   tf_random_global_seed, message):

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

    dataset_id = dataset_directory.split('/')[0]
    dataset_type = dataset_directory.split('/')[-1]

    local_dataset_dir = Path(tmp_directory, 'datasets')
    local_model_dir = Path(tmp_directory, 'models')

    copy_folder_locally_if_missing(os.path.join(gcp_bucket, 'datasets', dataset_directory),
                                   Path(local_dataset_dir, dataset_id))

    copy_file_locally_if_missing(os.path.join(gcp_bucket, 'datasets', dataset_id, 'config.yaml'),
                                 Path(local_dataset_dir, dataset_id, 'config.yaml'))

    copy_folder_locally_if_missing(os.path.join(gcp_bucket, 'models', model_id), local_model_dir)

    train_thresh_id = "{}_{}_{}".format(model_id, dataset_id, optimizing_class_metric)
    train_thresh_id_dir = Path(tmp_directory, str('train_thresholds_' + train_thresh_id))
    train_thresh_id_dir.mkdir(parents=True)

    with Path(local_dataset_dir, dataset_id, 'config.yaml').open('r') as f:
        dataset_config = yaml.safe_load(f)['dataset_config']

    with Path(local_model_dir, model_id, 'config.yaml').open('r') as f:
        train_config = yaml.safe_load(f)['train_config']

    target_size = dataset_config['target_size']

    if 'validation' in dataset_type:
        gen_seed = None if 'validation_data_shuffle_seed' not in train_config else train_config['validation_data_shuffle_seed']
    elif 'test' in dataset_type:
        gen_seed = None if 'test_data_shuffle_seed' not in train_config else train_config['test_data_shuffle_seed']
    else:
        gen_seed = 1234

    train_threshold_dataset_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, dataset_directory).as_posix(),
        rescale=1./255,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        seed=gen_seed)

    trained_prediction_thresholds = {}
    training_thresholds_output = {}
    opt_config = []
    for i in range(len(train_threshold_dataset_generator.mask_filenames)):
        print('\n' + str('Training class' + str(i) + ' prediction threshold...'))
        training_threshold_output, opt_config = train_prediction_thresholds(i, optimizing_class_metric, train_config,
                                                                            train_threshold_dataset_generator,
                                                                            dataset_downsample_factor,
                                                                            Path(local_model_dir, model_id,
                                                                                 "model.hdf5").as_posix())
        if not training_threshold_output.success:
            AssertionError("Training prediction thresholds has failed. See function minimization command line output.")

        training_thresholds_output.update({str('class'+str(i)): {'x': float(training_threshold_output.x),
                                                                 'success': training_threshold_output.success,
                                                                 'status': training_threshold_output.status,
                                                                 'message': training_threshold_output.message,
                                                                 'nfev': training_threshold_output.nfev,
                                                                 'fun': float(training_threshold_output.fun)}})
        trained_prediction_thresholds.update({str('class' + str(i)): float(training_threshold_output.x)})

    metadata = {
        'message': message,
        'gcp_bucket': gcp_bucket,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'num_classes': len(train_threshold_dataset_generator.mask_filenames),
        'target_size': target_size,
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1),
        'dataset_directory': dataset_directory,
        'model_id': model_id,
        'batch_size': batch_size,
        'dataset_config': dataset_config,
        'train_config': train_config,
        'thresholds_training_configuration': opt_config,
        'thresholds_training_output': training_thresholds_output,
        'thresholds_training_history': thresholds_training_history,
        'random-module-global-seed': random_module_global_seed,
        'numpy_random_global_seed': numpy_random_global_seed,
        'tf_random_global_seed': tf_random_global_seed
    }

    metadata_sys = {
        'System_info': getSystemInfo(),
        'Lib_versions_info': getLibVersions()
    }

    output_data = {
        'final_trained_prediction_thresholds': trained_prediction_thresholds,
        'metadata': metadata,
        'metadata_system': metadata_sys
    }

    with Path(train_thresh_id_dir, output_file_name).open('w') as f:
        yaml.safe_dump(output_data, f)

    # copy without overwrite
    os.system("gsutil -m cp -n -r '{}' '{}'".format(Path(train_thresh_id_dir, output_file_name).as_posix(),
                                                    os.path.join(gcp_bucket, 'models', model_id)))

    print('\n Train Prediction Thresholds Results:')
    print(trained_prediction_thresholds)

    print('\n Train Prediction Thresholds Metadata:')
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
        '--dataset-directory',
        type=str,
        help='The dataset ID + "/validation" or "/test".')
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
        '--optimizing-class-metric',
        type=str,
        default='iou_score',
        help='Use single class metric if training prediction threshold.')
    argparser.add_argument(
        '--dataset-downsample-factor',
        type=float,
        default=1.0,
        help='Accelerate optimization via using subset of dataset.')
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
        '--message',
        type=str,
        default=None,
        help='A str message the used wants to leave, the default is None.')
    train_segmentation_model_prediction_thresholds(**argparser.parse_args().__dict__)
