# necstlab Damage Segmentation

This repository contains the documentation and code to train and use a model for damage segmentation of carbon-fiber-laminate tomography images. For example, this is a properly annotated image:

![Annotated image](images/annotated_image.png)

## Workflows

To accomplish this, there are five workflows that this repository supports: 
* [**data ingestion**](docs/data_ingestion.md): copying the raw data into a cloud bucket and logically structuring it
* [**dataset preparation**](docs/dataset_preparation.md): preparing a dataset for use in training and testing
* [**training**](docs/training.md): training a damage segmentation model on a dataset
* [**testing**](docs/testing.md): testing the performance of a pretrained damage segmentation model on a dataset
* [**inference**](docs/inference.md): segmenting the damage of an image stack

Before running any of these workflows, you'll need to [set up your local machine](docs/local_setup.md) and [set up your GCP bucket](docs/gcp_bucket_setup.md). You may also want to look through [assumed knowledge](docs/assumed_knowledge.md).

Atention: For significant code changes to any files except `configs` and `.md`'s, users must:
1 Create new branch on github web browser
1 Refresh the local desktop client and switch to the new branch, 
1 Make the significant change in new branch
1 Commit change in local desktop client, 
1 Push commit to remote git (i.e., web browser) using local client
1 Create pull request in web browser with Josh Joseph as reviewer
1 Once approved, complete merge and then delete branch

# Known gotchas
* You can only run a single workflow at a time on a VM (due to different runs possibly stepping on each other through the temp directory. [#27](https://github.com/mit-quest/necstlab-damage-segmentation/issues/27) will address this.
* The U-model and parameters are hardcoded in train, test, and inference. So you'll need to change all if you want change any. [#3](https://github.com/mit-quest/necstlab-damage-segmentation/issues/3) will address this.
