pipenv run python3 prepare_dataset.py --config-file configs/dataset_config-small.yaml
pipenv run python3 train_segmentation_model.py --config-file configs/train_config-small.yaml
pipenv run python3 test_segmentation_model.py --config-file configs/test_config-small.yaml