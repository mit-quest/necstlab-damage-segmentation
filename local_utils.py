
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


def folder_has_files(directory, id):
    if os.listdir(directory) == []:
        raise FileNotFoundError('There are no files in ' + str(directory) + '. Confirm that ' + id + ' exists.')


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
