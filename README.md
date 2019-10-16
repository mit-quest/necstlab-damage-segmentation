# necstlab Damage Segmentation

This repository contains the documentation and code to train and use a model for damage segmentation of carbon-fiber-laminate scans (which we refer to as "image stacks"). To accomplish this, there are five workflows that this repository supports: 
* [**data ingestion**](docs/data_ingestion.md): copying the raw data into a cloud bucket and logically structuring it
* [**dataset preparation**](docs/dataset_preparation.md): preparing a dataset for use in training and testing
* [**training**](docs/training.md): training a damage segmentation model on a dataset
* [**testing**](docs/testing.md): testing the performance of a pretrained damage segmentation model on a dataset
* [**inference**](docs/inference.md): segmenting the damage of an image stack

Before running any of these workflows, you'll need to [set up your local machine](docs/local_setup.md) and [set up your GCP bucket](docs/gcp_bucket_setup.md). You may also want to look through [assumed knowledge](docs/assumed_knowledge.md).
