import os
from pathlib import Path
from google.cloud import storage
import platform
import socket
import psutil
import logging
import GPUtil
import sys
import tensorflow as tf
import os
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


def getSystemInfo():
    try:
        info = {}
        info['Platform'] = {}
        info['Platform']['name'] = platform.system()
        info['Platform']['release'] = platform.release()
        info['Platform']['version'] = platform.version()
        info['Hostname'] = socket.gethostname()

        info['Processor'] = platform.processor()
        info['Ram'] = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
        info['CPU'] = {}
        info['CPU']['number'] = psutil.cpu_count(logical=True)

        gpus = GPUtil.getGPUs()
        info['GPU'] = {}
        info['GPU']['number'] = len(gpus)
        gpu = gpus[0]  # assumes all are the same
        info['GPU']['name'] = gpu.name
        info['GPU']['memory'] = str(round(gpu.memoryTotal / (1024.0))) + " GB"
        return info
    except IOError:
        return None


def getLibVersions():
    try:
        info = {}
        info['python'] = sys.version
        info['tensorflow'] = tf.__version__
        info['segmentation_models'] = segmentation_models.__version__
        return info
    except IOError:
        return None
