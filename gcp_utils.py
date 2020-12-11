import os
from pathlib import Path
from google.cloud import storage
import tensorflow as tf
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models


def list_files(gcp_bucket_name, prefix):
    gcp_key_file_location = next(Path('./keys').iterdir())
    storage_client = storage.Client.from_service_account_json(gcp_key_file_location.as_posix())
    bucket_name = gcp_bucket_name
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix, max_results=10000)
    return [f.name for f in blobs]


def copy_folder_locally_if_missing(folder_remote_source, local_folder_dir):
    if not os.path.exists(local_folder_dir.as_posix()):
        local_folder_dir.mkdir(parents=True, exist_ok=True)
        os.system("gsutil -m cp -r '{}' '{}'".format(folder_remote_source, local_folder_dir.as_posix()))


def copy_file_locally_if_missing(file_remote_path, local_file_path):
    os.system("gsutil cp -n '{}' '{}'".format(file_remote_path, local_file_path))


def remote_folder_exists(remote_dest, folder_name, sample_file_name=None):
    with Path('terraform.tfvars').open() as f:
        line = f.readline()
        while line:
            if 'gcp_key_file_location' in line:
                gcp_key_file_location = line.split('"')[1]
            line = f.readline()

    storage_client = storage.Client.from_service_account_json(gcp_key_file_location)
    bucket_name = remote_dest.split('/')[2]
    bucket = storage_client.get_bucket(bucket_name)

    if sample_file_name is not None:
        folder_name = '/'.join([folder_name, sample_file_name])

    blobs = bucket.list_blobs(prefix='/'.join(remote_dest.split('/')[3:] + [folder_name]),
                              max_results=None)

    folder_exists = False
    for blob in blobs:
        if blob.name.split('/')[1] == folder_name.split('/')[0]:
            folder_exists = True
            break

    return folder_exists
