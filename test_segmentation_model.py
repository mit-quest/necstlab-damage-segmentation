import shutil
import os
import yaml
from pathlib import Path
from datetime import datetime
import pytz
import git
from gcp_utils import copy_folder_locally_if_missing
from image_utils import ImagesAndMasksGenerator
from models import generate_compiled_segmentation_model


metadata_file_name = 'metadata.yaml'
tmp_directory = Path('./tmp')


def test(gcp_bucket, dataset_id, model_id, batch_size):

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

    test_id = "{}_{}".format(model_id, dataset_id, datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'))
    test_dir = Path(tmp_directory, 'tests', test_id)
    test_dir.mkdir(parents=True)

    with Path(local_dataset_dir, dataset_id, 'config.yaml').open('r') as f:
        dataset_config = yaml.safe_load(f)['dataset_config']

    with Path(local_model_dir, model_id, 'config.yaml').open('r') as f:
        train_config = yaml.safe_load(f)['train_config']

    target_size = dataset_config['target_size']

    test_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, dataset_id, 'test').as_posix(),
        rescale=1./255,
        target_size=target_size,
        batch_size=batch_size,
        seed=None)  # train_config['test_data_shuffle_seed'])

    compiled_model = generate_compiled_segmentation_model(
        train_config['segmentation_model']['model_name'],
        train_config['segmentation_model']['model_parameters'],
        len(test_generator.mask_filenames),
        train_config['loss'],
        train_config['optimizer'],
        Path(local_model_dir, model_id, "model.hdf5").as_posix())

    # manually check metric memory location and inheritance chain
    # print(compiled_model.metrics)
    # for m in compiled_model.metrics:
    #     if hasattr(m.__class__, '__name__'):
    #         print(m.name, m.__class__.__name__, m.__class__.__mro__)
    # input("post-compile - press enter")

    results = compiled_model.evaluate(test_generator)

    if hasattr(compiled_model.loss, '__name__'):
        metric_names = [compiled_model.loss.__name__] + [m.name for m in compiled_model.metrics]
    elif hasattr(compiled_model.loss, 'name'):
        metric_names = [compiled_model.loss.name] + [m.name for m in compiled_model.metrics]

    with Path(test_dir, 'metrics.csv').open('w') as f:
        f.write(','.join(metric_names) + '\n')
        f.write(','.join(map(str, results)))

    metadata = {
        'gcp_bucket': gcp_bucket,
        'dataset_id': dataset_id,
        'model_id': model_id,
        'batch_size': batch_size,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1),
        'dataset_config': dataset_config,
        'train_config': train_config
    }

    with Path(test_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -r '{}' '{}'".format(Path(tmp_directory, 'tests').as_posix(), gcp_bucket))

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

    test(**argparser.parse_args().__dict__)
