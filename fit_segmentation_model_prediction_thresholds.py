import shutil
import os
import yaml
from pathlib import Path
from datetime import datetime
import pytz
import git
from gcp_utils import copy_folder_locally_if_missing, copy_file_locally_if_missing
from image_utils import ImagesAndMasksGenerator
from models import fit_prediction_thresholds

# can fit same model repeatedly with different optimization configurations
metadata_file_name = 'metadata_fit_thresholds_output_' + datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ') + '.yaml'
tmp_directory = Path('./tmp')


def fit_segmentation_model_prediction_thresholds(gcp_bucket, dataset_directory, model_id, batch_size,
                                                 optimizing_class_metric, dataset_downsample_factor):

    start_dt = datetime.now()

    assert "gs://" in gcp_bucket

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    dataset_id = dataset_directory.split('/')[0]

    local_dataset_dir = Path(tmp_directory, 'datasets')
    local_model_dir = Path(tmp_directory, 'models')

    copy_folder_locally_if_missing(os.path.join(gcp_bucket, 'datasets', dataset_directory),
                                    Path(local_dataset_dir, dataset_id))

    copy_file_locally_if_missing(os.path.join(gcp_bucket, 'datasets', dataset_id, 'config.yaml'),
                                 Path(local_dataset_dir, dataset_id, 'config.yaml'))

    copy_folder_locally_if_missing(os.path.join(gcp_bucket, 'models', model_id), local_model_dir)

    fit_id = "{}_{}_{}".format(model_id, dataset_id, optimizing_class_metric)
    fit_id_dir = Path(tmp_directory, str('fit_thresholds_' + fit_id))
    fit_id_dir.mkdir(parents=True)

    with Path(local_dataset_dir, dataset_id, 'config.yaml').open('r') as f:
        dataset_config = yaml.safe_load(f)['dataset_config']

    with Path(local_model_dir, model_id, 'config.yaml').open('r') as f:
        train_config = yaml.safe_load(f)['train_config']

    target_size = dataset_config['target_size']

    fit_dataset_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, dataset_directory).as_posix(),
        rescale=1./255,
        target_size=target_size,
        batch_size=batch_size,
        seed=None if 'validation_data_shuffle_seed' not in train_config else train_config['validation_data_shuffle_seed'])

    prediction_thresholds_optimized = {}
    opt_config = []
    for i in range(len(fit_dataset_generator.mask_filenames)):
        print('\n' + str('Fitting class ' + str(i) + ' prediction threshold...'))
        prediction_threshold_optimized, opt_config = fit_prediction_thresholds(i, optimizing_class_metric, train_config,
                                                                               fit_dataset_generator,
                                                                               dataset_downsample_factor,
                                                                               Path(local_model_dir, model_id,
                                                                                    "model.hdf5").as_posix())
        prediction_thresholds_optimized.update({str('class_'+str(i)): {'x': float(prediction_threshold_optimized.x),
                                                                       'success': prediction_threshold_optimized.success,
                                                                       'status': prediction_threshold_optimized.status,
                                                                       'message': prediction_threshold_optimized.message,
                                                                       'nfev': prediction_threshold_optimized.nfev,
                                                                       'fun': float(prediction_threshold_optimized.fun)}})

    metadata = {
        'gcp_bucket': gcp_bucket,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'num_classes': len(fit_dataset_generator.mask_filenames),
        'target_size': target_size,
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1),
        'dataset_directory': dataset_directory,
        'model_id': model_id,
        'batch_size': batch_size,
        'dataset_config': dataset_config,
        'train_config': train_config,
        'threshold_optimization_configuration': opt_config,
        'prediction_thresholds_optimized': prediction_thresholds_optimized
    }

    with Path(fit_id_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    # copy without overwrite
    os.system("gsutil -m cp -n -r '{}' '{}'".format(Path(fit_id_dir).as_posix(),
                                                    os.path.join(gcp_bucket, 'models', model_id)))

    print('\n Fit Prediction Thresholds Metadata:')
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
        default='iou_score_1H',
        help='Use single class metric if fitting prediction threshold.')
    argparser.add_argument(
        '--dataset-downsample-factor',
        type=float,
        default=1.0,
        help='Accelerate optimization via using subset of dataset.')

    fit_segmentation_model_prediction_thresholds(**argparser.parse_args().__dict__)
