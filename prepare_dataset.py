"""
The ordering of the steps is important because it assumes a certain directory structure is progressively created!
"""

import os
import shutil
import random
import math
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from google.cloud import storage
import git
from datetime import datetime
import pytz


ply_GV_mapping = {
    'no_damage': 0,
    '0-degree_damage': 100,
    '45-degree_damage': 175,
    '90-degree_damage': 250
}

metadata_file_name = 'metadata.yaml'
tmp_directory = Path('./tmp')


def remote_dataset_exists(prepared_dataset_remote_dest, dataset_name):

    with Path('terraform.tfvars').open() as f:
        line = f.readline()
        while line:
            if 'gcp_key_file_location' in line:
                gcp_key_file_location = line.split('"')[1]
            line = f.readline()

    storage_client = storage.Client.from_service_account_json(gcp_key_file_location)
    bucket_name = prepared_dataset_remote_dest.split('/')[2]
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix='/'.join(prepared_dataset_remote_dest.split('/')[3:] + [dataset_name]),
                              max_results=1)
    return len(list(blobs)) >= 1


def copy_processed_data_locally_if_missing(scans, processed_data_remote_source, processed_data_local_dir):
    Path(processed_data_local_dir).mkdir(parents=True, exist_ok=True)
    for scan in scans:
        if scan not in [p.name for p in Path(processed_data_local_dir).iterdir()]:
            os.system("gsutil -m cp -r '{}/{}' '{}'".format(processed_data_remote_source, scan, processed_data_local_dir))


def copy_and_downsample_processed_data_to_preparation_if_missing(scans, processed_data_local_dir,
                                                                 data_prep_local_dir, downsampling_params):
    Path(data_prep_local_dir, 'downsampled').mkdir(parents=True, exist_ok=True)
    for scan in scans:
        if scan not in [p.name for p in Path(data_prep_local_dir, 'downsampled').iterdir()]:
            scan_image_files = sorted(Path(processed_data_local_dir, scan, 'images').iterdir())
            scan_annotation_files = sorted(Path(processed_data_local_dir, scan, 'annotations').iterdir())
            assert len(scan_image_files) == len(scan_annotation_files)
            total_images = len(scan_image_files)

            if 'number_of_images' in downsampling_params:
                num_images = downsampling_params['number_of_images']
            elif 'frac' in downsampling_params:
                num_images = math.ceil(downsampling_params['frac'] * total_images)
            else:
                num_images = total_images

            if downsampling_params['type'] == 'None':
                file_inds_to_copy = range(0, total_images)
            elif downsampling_params['type'] == 'random':
                file_inds_to_copy = random.sample(range(0, total_images), k=num_images)
            elif downsampling_params['type'] == 'linear':
                file_inds_to_copy = np.floor(np.linspace(0, total_images-1, num_images)).astype(int)
            elif downsampling_params['type'] == 'from_start':
                file_inds_to_copy = range(0, num_images)
            elif downsampling_params['type'] == 'from_end':
                file_inds_to_copy = range(total_images - num_images, total_images)
            else:
                raise ValueError("Unknown downsampling type: {}".format(downsampling_params['type']))

            Path(data_prep_local_dir, 'downsampled', scan, 'images').mkdir(parents=True)
            Path(data_prep_local_dir, 'downsampled', scan, 'annotations').mkdir(parents=True)
            for image_ind in file_inds_to_copy:
                shutil.copy(scan_image_files[image_ind].as_posix(),
                            Path(data_prep_local_dir, 'downsampled', scan, 'images', scan_image_files[
                                image_ind].name).as_posix())
                shutil.copy(scan_annotation_files[image_ind].as_posix(),
                            Path(data_prep_local_dir, 'downsampled', scan, 'annotations', scan_annotation_files[
                                image_ind].name).as_posix())


# adapted from: https://github.com/matterport/Mask_RCNN/issues/230
def random_crop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask


def resize_and_crop(data_prep_local_dir, target_size, image_cropping_params):
    Path(data_prep_local_dir, 'resized').mkdir(parents=True, exist_ok=True)
    for scan in [p.name for p in Path(data_prep_local_dir, 'downsampled').iterdir()]:
        if scan not in [p.name for p in Path(data_prep_local_dir, 'resized').iterdir()]:
            scan_image_files = sorted(Path(data_prep_local_dir, 'downsampled', scan, 'images').iterdir())
            scan_annotation_files = sorted(Path(data_prep_local_dir, 'downsampled', scan, 'annotations').iterdir())
            assert len(scan_image_files) == len(scan_annotation_files)
            Path(data_prep_local_dir, 'resized', scan, 'images').mkdir(parents=True)
            Path(data_prep_local_dir, 'resized', scan, 'annotations').mkdir(parents=True)
            for image_ind in range(len(scan_image_files)):
                image = Image.open(scan_image_files[image_ind])
                annotation = Image.open(scan_annotation_files[image_ind])
                if image_cropping_params['type'] == 'None':
                    image.thumbnail(target_size)
                    annotation.thumbnail(target_size)
                elif image_cropping_params['type'] == 'random':
                    assert image_cropping_params['num_per_image'] == 1
                    image, annotation = random_crop(np.asarray(image), np.asarray(annotation), target_size[0], target_size[1])
                    image = Image.fromarray(image)
                    annotation = Image.fromarray(annotation)
                else:
                    raise ValueError("Image cropping type: {}".format(image_cropping_params['type']))
                image.save(Path(data_prep_local_dir, 'resized', scan, 'images', scan_image_files[
                    image_ind].name).as_posix())
                annotation.save(Path(data_prep_local_dir, 'resized', scan, 'annotations', scan_annotation_files[
                    image_ind].name).as_posix())


def create_class_masks(data_prep_local_dir, class_annotation_mapping):
    for scan in [p.name for p in Path(data_prep_local_dir, 'resized').iterdir()]:
        if 'masks' not in [p.name for p in Path(data_prep_local_dir, 'resized', scan).iterdir()]:
            scan_annotation_files = sorted(Path(data_prep_local_dir, 'resized', scan, 'annotations').iterdir())
            for c, damage_plies_in_c in class_annotation_mapping.items():
                Path(data_prep_local_dir, 'resized', scan, 'masks', c).mkdir(parents=True, exist_ok=True)
                for scan_annotation_file in scan_annotation_files:
                    annotation = np.asarray(Image.open(scan_annotation_file))
                    mask = np.zeros(annotation.shape, dtype=bool)
                    for damage_ply in damage_plies_in_c:
                        mask += (annotation == ply_GV_mapping[damage_ply])
                    mask = Image.fromarray(mask.astype('uint8'))
                    mask.save(Path(data_prep_local_dir, 'resized', scan, 'masks', c, scan_annotation_file.name).as_posix())


def recursive_copy_directory(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError:
        shutil.copy(src, dst)


def split_prepared_data(data_prep_local_dir, prepared_dataset_local_dir, dataset_split):
    for split, scans_in_split in dataset_split.items():
        if split not in [p.name for p in prepared_dataset_local_dir.iterdir()]:
            images_dest_dir = Path(prepared_dataset_local_dir, split, 'images')
            images_dest_dir.mkdir(parents=True)
            for scan in scans_in_split:
                for file in Path(data_prep_local_dir, 'resized', scan, 'images').iterdir():
                    shutil.copy(file.as_posix(), Path(images_dest_dir, file.name).as_posix())
                for c in [p.name for p in Path(data_prep_local_dir, 'resized', scan, 'masks').iterdir()]:
                    mask_class_dest_dir = Path(prepared_dataset_local_dir, split, 'masks', c)
                    mask_class_dest_dir.mkdir(parents=True, exist_ok=True)
                    for file in Path(data_prep_local_dir, 'resized', scan, 'masks', c).iterdir():
                        shutil.copy(file.as_posix(), Path(mask_class_dest_dir, file.name).as_posix())


def copy_dataset_to_remote_dest(prepared_dataset_location, prepared_dataset_remote_dest, dataset_id):
    os.system("gsutil -m cp -r '{}' '{}'".format(prepared_dataset_location.as_posix(),
                                                 os.path.join(prepared_dataset_remote_dest, dataset_id)))


def prepare_dataset(gcp_bucket, config_file):

    start_dt = datetime.now()

    with Path(config_file).open('r') as f:
        dataset_config = yaml.safe_load(f)['dataset_config']

    dataset_id = Path(config_file).name.split('.')[0]

    assert "gs://" in gcp_bucket

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    processed_data_remote_source = os.path.join(gcp_bucket, 'processed-data')
    processed_data_local_dir = Path(tmp_directory, 'processed-data')
    processed_data_local_dir.mkdir()

    data_prep_local_dir = Path(tmp_directory, 'preparing')
    data_prep_local_dir.mkdir()

    prepared_dataset_local_dir = Path(tmp_directory, 'datasets', )
    prepared_dataset_local_dir.mkdir(parents=True)

    prepared_dataset_remote_dest = os.path.join(gcp_bucket, 'datasets')

    with Path(prepared_dataset_local_dir, 'config.yaml').open('w') as f:
        yaml.safe_dump({'dataset_config': dataset_config}, f)

    all_scans = []
    for _, scans in dataset_config['dataset_split'].items():
        all_scans += scans
    all_scans = sorted(set(all_scans))

    assert not remote_dataset_exists(prepared_dataset_remote_dest, dataset_id)

    copy_processed_data_locally_if_missing(all_scans, processed_data_remote_source, processed_data_local_dir)

    copy_and_downsample_processed_data_to_preparation_if_missing(
        all_scans, processed_data_local_dir, data_prep_local_dir, dataset_config['stack_downsampling'])

    resize_and_crop(data_prep_local_dir, dataset_config['target_size'], dataset_config['image_cropping'])

    create_class_masks(data_prep_local_dir, dataset_config['class_annotation_mapping'])

    split_prepared_data(data_prep_local_dir, prepared_dataset_local_dir, dataset_config['dataset_split'])

    metadata = {
        'gcp_bucket': gcp_bucket,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'number_of_images': {
            'train': len(list(Path(prepared_dataset_local_dir, 'train', 'images').iterdir())),
            'validation': len(list(Path(prepared_dataset_local_dir, 'validation', 'images').iterdir())),
        },
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'original_config_filename': config_file,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1)
    }
    try:
        metadata['number_of_images']['test'] = len(list(Path(prepared_dataset_local_dir, 'test', 'images').iterdir()))
    except FileNotFoundError:
        pass  # does not necessarily have to be test data

    with Path(prepared_dataset_local_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    copy_dataset_to_remote_dest(prepared_dataset_local_dir, prepared_dataset_remote_dest, dataset_id)

    shutil.rmtree(tmp_directory.as_posix())


if __name__ == "__main__":
    import argparse
    import sys

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument(
        '--gcp-bucket',
        type=str,
        help='The GCP bucket where the processed data is located and to use to store the prepared dataset.')
    argparser.add_argument(
        '--config-file',
        type=str,
        help='The location of the data preparation configuration file.')

    prepare_dataset(**argparser.parse_args().__dict__)
