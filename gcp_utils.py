import os
from pathlib import Path
from google.cloud import storage


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
