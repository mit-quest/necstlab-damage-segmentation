# necstlab's Damage Segmentation

This repository contains the instructions and code to train and use a model for damage segmentation of carbon-fiber-laminate scans (which are often referred to as "image stacks"). To accomplish this, there are five workflows that this repository supports: 
* copying the raw data into the cloud for storage and usage
* segmenting the damage of an image stack (often called inference)
* preparing a dataset for training and testing of a damage segmentation model
* testing the performance of a pretrained damage segmentation model on a dataset
* training a damage segmentation model on a dataset

## Prerequisites for all workflows

### Setting up your local machine

#### Clone this repository

In a terminal window, enter `git clone git@github.com:mit-quest/necstlab-damage-segmentation.git`.

Change into the main directory: `cd necstlab-damage-segmentation`

If you do not have `git` installed, see [here]() for installation instructions. TODO: add link/link content

#### Terraform

To programmatically set up and destroy cloud resources (virtual machines, buckets, etc.), we will use a tool called Terraform. For instructions on how to install Terraform, see [here](). TODO: add link/link content

#### GCP

All of the workflows use Google Cloud Platform (GCP) for storage (buckets) and compute (virtual machines). To allow the code to programmatically interact with GCP, we will set up a Software Development Kit (SDK) on your local machine. To install the GCP SDK follow the instructions [here](). TODO: add link/link content

To set up and destroy virtual machines, Terraform requires access to GCP. For instructions on how to download GCP credentials for Terraform, see [here](). TODO: add link/link content

#### Pipenv

For locally run python code we will use a tool called `pipenv` to set up and manage python virtual environments and packages. See the instructions [here]() for on how to install `pipenv`. TODO: add link/link content

Run the command `pipenv install` to set up the virtual environment and download the required python packages for the workflows.

### Setting up your GCP bucket

To store all of the artifacts (data, models, results, etc.) that are required for and result from the workflows, we will use a GCP bucket. To set up a GCP bucket to be used for these workflows, see the instructions [here](). TODO: add link/link content

As you run the workflows you'll see the following directory structure be automatically created and populated inside of your bucket:
```
<GCP bucket>/
    analyses/      (this is where any analysis that results from testing will be stored)
        <analysis_ID>-<timestamp>/
            plots/
                ...
            config.yaml
            log.txt
            metrics.csv
    raw-data/      (this is where any raw data will be stored)
        <stack_ID>/
            annotations/
                ...
            images/
                ...
            metadata.yaml
    datasets/      (this is where any prepared datasets for training will be stored)
        <dataset_ID>-<timestamp>/
            test/
                annotations/
                    ...
                images/
                    ...
            train/
                annotations/
                    ...
                images/
                    ...
            validate/
                annotations/
                    ...
                images/
                    ...
            config.yaml
            log.txt
    inferences/    (this is where any stack segmentations will be stored)
        <inference_ID>-<timestamp>/
            output/
                ...
            config.yaml
            log.txt
    models/        (this is where any trained segmentation models will be stored)
        <model_ID>-<timestamp>/
            plots/
                ...
            config.yaml
            log.txt
            model.hdf5
            metrics.csv
```
where `<analysis_ID>`, `<dataset_ID>`, `<inference_ID>`, and `<model_ID>` are defined inside of configuration files and `<timestamp>` is the time the process was started and is automatically generated. The `<stack_ID>`s will be created as raw data is moved into the cloud. 


### Assumed knowledge

The following workflows assume:
* You know how to check the status of GCP virtual machines using the GCP compute engine dashboard. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content 
* You know how to SSH into a GCP virtual machine. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content
* You know how to check the contents of a GCP bucket using the GCP storage dashboard. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content 
* You know how to create and destroy resources using Terraform. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content 
* You are familiar with image annotations and how they are used in image segmentation. If you are unfamiliar with this, see [here]() for more information. TODO: add link/link content 
* You are familiar with how datasets are used in Machine Learning (for example, splitting your data into train, validation, and test). If you are unfamiliar with this, see [here]() for more information. TODO: add link/link content  

## Copying the raw data into the cloud for storage and usage

Prerequisite artifacts:
* Stacks of images and annotations that we wish to use in the other workflows on your local machine

Infrastructure that will be used:
* A GCP bucket where the stacks will be stored
* Your local machine to upload the stacks to the GCP bucket

### Workflow

1. To copy the unsegmented stacks to a GCP bucket run the command: `pipenv run python3 ingest_data_to_gcp.py --local-stacks-dir <local_stacks_dir> --gcp-bucket <gcp_bucket>` where `<local_stacks_dir>` is the local directory where the stacks are stored and `<gcp_bucket>` is the bucket where our artifacts will be stored. 
1. When this completes, you should see all of your stacks in `<gcp_bucket>/data/ingested/<stack_ID>` where `<stack_ID>` are the names of the directories inside of `<local_stacks_dir>`.

## Segmenting damage of a set of stacks

Prerequisite artifacts:
* Unannotated stacks (in a GCP bucket) that we wish to perform damage segmentation on
* A pretrained damage segmentation model (in a GCP bucket) to use for inference
* An inference configuration file (on your local machine)

Infrastructure that will be used:
* A GCP bucket where the stored unsegmented stacks will be accessed from
* A GCP bucket where the stacks with inferred damage segmentation will be stored
* A GCP virtual machine to run inference on

### Workflow

1. If the unsegmented stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). 
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To infer (segment) the damage of the stacks, SSH into the virtual machine `<project_name>-<user_name>` and run `pipenv run python3 infer_segmentation.py --config-file configurations/inference.yaml`. 
1. Once inference has finished, you should see the folder `<gcp_bucket>/inferences/<inference_ID>-<timestamp>` has been created and populated, where `<inference_ID>` was defined in `configurations/inference.yaml`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

## Preparing a dataset for training and testing of a segmentation model 

Prerequisite artifacts:
* Annotated stacks (in a GCP bucket) that we will use to create the dataset
* A dataset configuration file (on your local machine)

Infrastructure that will be used:
* A GCP bucket where the segmented stacks will be accessed from
* A GCP bucket where the prepared dataset will be stored
* A GCP virtual machine to run the training on

### Workflow

1. If the stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). 
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To create a dataset, SSH into the virtual machine `<project_name>-<user_name>` and run `pipenv run python3 prepare_data.py --config-file configurations/data_preparation.yaml`. 
1. Once dataset preparation has finished, you should see the folder `<gcp_bucket>/datasets/<dataset_ID>-<timestamp>` has been created and populated, where `<dataset_ID>` was defined in `configurations/data_preparation.yaml`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

## Testing a segmentation model

Prerequisite artifacts:
* A dataset (in a GCP bucket) that we will use to test the performance of the model against 
* A pretrained damage segmentation model (in a GCP bucket) to test the performance of
* A test configuration files (on your local machine)

Infrastructure that will be used:
* A GCP bucket where the segmented stacks will be accessed from
* A GCP bucket where the results of the analysis will be stored
* A GCP virtual machine to run the analysis on

### Workflow
1. If the stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). 
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To create a dataset, SSH into the virtual machine `<project_name>-<user_name>` and run `pipenv run python3 test_segmentation_model.py --config-file configurations/test.yaml`. 
1. Once dataset preparation has finished, you should see the folder `<gcp_bucket>/analyses/<analysis_ID>-<timestamp>` has been created and populated, where `<analysis_ID>` was defined in `configurations/test.yaml`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed. 

## Training a segmentation model

Prerequisite artifacts:
* A dataset (in a GCP bucket) that we will use to train the model
* A train configuration file (on your local machine)

Infrastructure that will be used:
* A GCP bucket where the prepared dataset for training will be accessed from
* A GCP bucket where the trained model will be stored
* A GCP virtual machine to run the training on

### Workflow
1. If the stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). 
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To create a dataset, SSH into the virtual machine `<project_name>-<user_name>` and run `pipenv run python3 train_segmentation_model.py --config-file configurations/train.yaml`. 
1. Once dataset preparation has finished, you should see the folder `<gcp_bucket>/models/<model_ID>-<timestamp>` has been created and populated, where `<model_ID>` was defined in `configurations/train.yaml`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed. 
