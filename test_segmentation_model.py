import shutil
import os
import csv
import yaml
from pathlib import Path
from datetime import datetime
import pytz
from keras.optimizers import Adam
from segmentation_models import Unet
from segmentation_models.metrics import iou_score
import git
from gcp_utils import copy_dataset_locally_if_missing
from image_utils import ImagesAndMasksGenerator


metadata_file_name = 'metadata.yaml'
tmp_directory = Path('./tmp')


def test(config_file):

    start_dt = datetime.now()

    with Path(config_file).open('r') as f:
        test_config = yaml.safe_load(f)['test_config']

    assert "gs://" in test_config['gcp_bucket']

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    local_dataset_dir = Path(tmp_directory, 'datasets')
    local_model_dir = Path(tmp_directory, 'models')

    copy_dataset_locally_if_missing(os.path.join(test_config['gcp_bucket'], 'datasets', test_config['dataset_id']),
                                    local_dataset_dir)

    copy_dataset_locally_if_missing(os.path.join(test_config['gcp_bucket'], 'models', test_config['model_id']),
                                    local_model_dir)

    test_id = "{}_{}".format(test_config['model_id'], test_config['dataset_id'],
                             datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'))
    test_dir = Path(tmp_directory, 'tests', test_id)
    test_dir.mkdir(parents=True)

    with Path(local_dataset_dir, test_config['dataset_id'], 'config.yaml').open('r') as f:
        dataset_config = yaml.safe_load(f)['dataset_config']

    with Path(test_dir, 'config.yaml').open('w') as f:
        yaml.safe_dump({'train_config': dataset_config}, f)

    target_size = dataset_config['target_size']
    batch_size = test_config['batch_size']

    test_generator = ImagesAndMasksGenerator(
        Path(local_dataset_dir, test_config['dataset_id'],
             'test').as_posix(),
        rescale=1./255,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        seed=None)

    model = Unet('vgg16', input_shape=(None, None, 1), classes=len(test_generator.mask_filenames), encoder_weights=None)

    loss_fn = 'binary_crossentropy' if len(test_generator.mask_filenames) == 1 else 'categorical_crossentropy'

    model.compile(optimizer=Adam(),
                  loss=loss_fn,
                  metrics=["accuracy", iou_score])

    model.load_weights(Path(local_model_dir, "model.hdf5").as_posix())

    results = model.evaluate_generator(test_generator)

    with Path(test_dir, 'metrics.csv').open('w') as f:
        w = csv.DictWriter(f, results.history.keys())
        w.writeheader()
        w.writerow(results.history)

    metadata = {
        'gcp_bucket': test_config['gcp_bucket'],
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'original_config_filename': config_file,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1)
    }

    with Path(test_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -r '{}' '{}'".format(Path(tmp_directory, 'models').as_posix(),
                                                 test_config['gcp_bucket']))

    shutil.rmtree(tmp_directory.as_posix())


if __name__ == "__main__":
    import argparse
    import sys

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument(
        '--config-file',
        type=str,
        help='The location of the test configuration file.')

    test(**argparser.parse_args().__dict__)
