#!/bin/bash


pipenv run python3 ingest_raw_data.py --gcp-bucket gs://necstlab-sandbox
pipenv run python3 prepare_dataset.py --gcp-bucket gs://necstlab-sandbox --config-file configs/dataset-large.yaml
pipenv run python3 train_segmentation_model.py --gcp-bucket gs://necstlab-sandbox --config-file configs/train-large.yaml
pipenv run python3 test_segmentation_model.py --gcp-bucket gs://necstlab-sandbox --dataset-id dataset-large --model-id segmentation-model-large_20190924T180419Z
pipenv run python3 infer_segmentation.py --gcp-bucket gs://necstlab-sandbox --model-id segmentation-model-large_20190924T180419Z --stack-id THIN_REF_S2_P1_L3_2496_1563_2159
pipenv run python3 infer_segmentation.py --gcp-bucket gs://necstlab-sandbox --model-id segmentation-model-large_20190924T180419Z --stack-id 8bit_AS4_S2_P1_L6_2560_1750_2160