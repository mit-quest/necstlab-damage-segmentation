#!/bin/bash


pipenv run python3 ingest_raw_data.py --gcp-bucket gs://necstlab-sandbox
pipenv run python3 prepare_dataset.py --gcp-bucket gs://necstlab-sandbox --config-file configs/dataset-small.yaml
pipenv run python3 train_segmentation_model.py --gcp-bucket gs://necstlab-sandbox --config-file configs/train-small.yaml
pipenv run python3 test_segmentation_model.py --gcp-bucket gs://necstlab-sandbox --dataset-id dataset-small --model-id segmentation-model-small_20190924T191717Z
pipenv run python3 infer_segmentation.py --gcp-bucket gs://necstlab-sandbox --model-id segmentation-model-small_20190924T191717Z --stack-id THIN_REF_S2_P1_L3_2496_1563_2159
pipenv run python3 infer_segmentation.py --gcp-bucket gs://necstlab-sandbox --model-id segmentation-model-small_20190924T191717Z --stack-id 8bit_AS4_S2_P1_L6_2560_1750_2160