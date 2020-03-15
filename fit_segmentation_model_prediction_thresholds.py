import shutil
import os
import yaml
from pathlib import Path
from datetime import datetime
import pytz
import git
from gcp_utils import copy_folder_locally_if_missing, copy_folder_locally_if_missing2
from image_utils import ImagesAndMasksGenerator
from models import fit_prediction_thresholds


metadata_file_name = 'metadata_fit_thresholds_' + datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ') + '.yaml'
tmp_directory = Path('./tmp')


def fit_segmentation_model_prediction_thresholds(gcp_bucket, dataset_id, model_id, batch_size,
                                                 optimizing_class_metric, dataset_downsample_factor):

    start_dt = datetime.now()

    assert "gs://" in gcp_bucket

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    local_dataset_dir = Path(tmp_directory, dataset_id)
    local_model_dir = Path(tmp_directory, 'models')

    copy_folder_locally_if_missing(os.path.join(gcp_bucket, 'datasets', dataset_id), local_dataset_dir, dataset_id)

    copy_folder_locally_if_missing2(os.path.join(gcp_bucket, 'models', model_id), local_model_dir)

    fit_id = "{}_{}".format(model_id, dataset_id, optimizing_class_metric,
                            datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'))
    fit_id_dir = Path(tmp_directory, 'fit_thresholds', fit_id)
    fit_id_dir.mkdir(parents=True)

    with Path(local_dataset_dir, dataset_id, 'config.yaml').open('r') as f:
        dataset_config = yaml.safe_load(f)['dataset_config']

    with Path(local_model_dir, model_id, 'config.yaml').open('r') as f:
        train_config = yaml.safe_load(f)['train_config']

    with Path(local_model_dir, model_id, metadata_file_name).open('r') as f:
        model_metadata = yaml.safe_load(f)

    target_size = dataset_config['target_size']

    fit_dataset_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, train_config['dataset_id'],
             'validation').as_posix(),
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
                                                                               Path(local_model_dir,
                                                                                    "model.hdf5").as_posix())
        prediction_thresholds_optimized.update({str('class_'+str(i)): {'x': float(prediction_threshold_optimized.x),
                                                                       'success': prediction_threshold_optimized.success,
                                                                       'status': prediction_threshold_optimized.status,
                                                                       'message': prediction_threshold_optimized.message,
                                                                       'nfev': prediction_threshold_optimized.nfev,
                                                                       'fun': float(prediction_threshold_optimized.fun)}})
        print(prediction_thresholds_optimized[str('class_'+str(i))])
        print(prediction_thresholds_optimized[str('class_' + str(i))]['x'])
        print(prediction_thresholds_optimized[str('class_' + str(i))]['fun'])
        input('enter')

    metadata = {
        'gcp_bucket': gcp_bucket,
        'dataset_id': dataset_id,
        'model_id': model_id,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1),
        'train_config': train_config,
        'threshold_optimization_configuration': opt_config,
        'prediction_thresholds_optimized': prediction_thresholds_optimized,
        'training_val_metadata': model_metadata
    }

    with Path(fit_id_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -n -r '{}' '{}'".format(Path(tmp_directory, 'fit_thresholds').as_posix(),
                                                    os.path.join(gcp_bucket, 'models', model_id)))

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
