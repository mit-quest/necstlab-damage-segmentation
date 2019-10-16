# necstlab's Damage Segmentation

This repository contains the instructions and code to train and use a model for damage segmentation of carbon-fiber-laminate scans (which we refer to as "image stacks"). To accomplish this, there are five workflows that this repository supports: 
* **data ingestion**: copying the raw data into a cloud bucket and logically structuring it
* **inference**: segmenting the damage of an image stack
* **dataset preparation**: preparing a dataset for use in training and testing
* **testing**: testing the performance of a pretrained damage segmentation model on a dataset
* **training**: training a damage segmentation model on a dataset

## Prerequisites for all workflows

### Setting up your local machine

#### Terraform

To programmatically set up and destroy cloud resources (virtual machines, buckets, etc.), we will use a tool called Terraform. For instructions on how to install Terraform, see [here](). TODO: add link/link content

#### GCP

All of the workflows use Google Cloud Platform (GCP) for storage (buckets) and compute (virtual machines). To allow the code to programmatically interact with GCP, we will set up a Software Development Kit (SDK) on your local machine. To install the GCP SDK follow the instructions [here](). TODO: add link/link content

To set up and destroy virtual machines, Terraform requires access to GCP. For instructions on how to download GCP credentials for Terraform, see [here](). TODO: add link/link content

Edit the `terraform.tfvars` file with your `username`, `gcp_key_file_location`, `public_ssh_key_location`, `private_ssh_key_location`. For more information on how to generate a public and private SSH key pair, see [here](https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

#### Set up this repository

To copy this repository locally, in a terminal window, enter and clone the repository using the command: `git clone git@github.com:mit-quest/necstlab-damage-segmentation.git`. If you do not have `git` installed, see [here]() for installation instructions. TODO: add link/link content

All commands will assume to be run from the `necstlab-damage-segmentation` directory, which you can `cd` into using: `cd necstlab-damage-segmentation`

### Setting up your GCP bucket

To store all of the artifacts (data, models, results, etc.) that are required for and result from the workflows, we will use a GCP bucket. To set up a GCP bucket to be used for these workflows, see the instructions [here](). TODO: add link/link content

As you run the workflows you'll see the following directory structure be automatically created and populated inside of your bucket:
```
<GCP bucket>/
    datasets/         (this is where any prepared datasets for training will be stored)
        <dataset_ID>/
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
            metadata.yaml
    inferences/       (this is where any stack segmentations will be stored)
        <inference_ID>-<timestamp>/
            logs/
            output/
                ...
            metadata.yaml
    models/           (this is where any trained segmentation models will be stored)
        <model_ID>-<timestamp>/
            logs/
            plots/
                ...
            config.yaml
            model.hdf5
            metadata.yaml
            metrics.csv
    processed-data/
        <stack_ID>/
            annotations/
                ...
            images/
                ...
            config.yaml
            metadata.yaml
    raw-data/         (this is where any raw data files will be stored)
        ...
    tests/         (this is where any analysis that results from testing will be stored)
        <test_ID>-<timestamp>/
            logs/
            plots/
                ...
            config.yaml
            metrics.csv
            metadata.yaml
```
where `<test_ID>`, `<dataset_ID>`, `<inference_ID>`, and `<model_ID>` are defined inside of configuration files and `<timestamp>` is the time the process was started and is automatically generated. The `<stack_ID>`s will be created as raw data is moved into the cloud. 


### Assumed knowledge

The following workflows assume:
* You know how to check the status of GCP virtual machines using the GCP compute engine dashboard. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content 
* You know how to SSH into a GCP virtual machine. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content
* You know how to check the contents of a GCP bucket using the GCP storage dashboard. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content 
* You know how to create and destroy resources using Terraform. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content 
* You are familiar with image annotations and how they are used in image segmentation. If you are unfamiliar with this, see [here]() for more information. TODO: add link/link content 
* You are familiar with how datasets are used in Machine Learning (for example, splitting your data into train, validation, and test). If you are unfamiliar with this, see [here]() for more information. TODO: add link/link content  
* You are familiar with how use tmux on a remote machine and how we will use it to keep processes running even if the SSH window is closed or disconnected. If you are unfamiliar with this, see [here]() for more information. TODO: add link/link content  

## Ingesting the raw data into a cloud bucket and processing it for usage in a dataset

Prerequisite artifacts:
* A stack of (zipped) images or annotations (tifs) that we wish to use in the other workflows on your local machine
* The zip filename is expected to look like:
    ``` 
    <stack_id>.zip (for unsegmented images)
    <stack_id>_dmg_labels_GV.zip (for annotations)    
    ```

Infrastructure that will be used:
* A GCP bucket where the raw and processed stacks will be stored
* Your local machine to upload the raw zip files to a GCP bucket
* A GCP machine to process the raw data

### Workflow

1. Copy the zip files of raw data into a GCP bucket: `gsutil -m cp <local_data_file> gs://<gcp_bucket_name>/raw-data` where `<gcp_bucket_name>` is the bucket where our artifacts will be stored. To copy an entire folder of zip files, `cd` into the directory and use the command: `gsutil -m cp -r . gs://<gcp_bucket_name>/raw-data`
1. When this completes, you should see your stack in `gs://<gcp_bucket_name>/raw-data/<zip_file>`.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). 
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. SSH into the GCP virtual machine, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and process a zip file by running the command: `pipenv run python3 process_raw_data.py --gcp-bucket gs://<gcp_bucket_name> --zipped-stack gs://<gcp_bucket_name>/raw-data/<zip_file>`. If the `--zipped-stack` argument is not given, it will process all of the files in `gs://<gcp_bucket_name>/raw-data`.
1. When this completes, you should see your stack in `gs://<gcp_bucket_name>/processed-data/<stack_ID>`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

Note: 
* Inside of `ingest_data_to_gcp.py` is stack-specific fixes to naming errors. You will have to edit that file for new scans with naming errors.
* instead of using the `--zipped-stack` argument, `--zips-dir` can be used instead with `ingest_data_to_gcp.py` to ingest a directory of zip files. 

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
1. Either edit the configuration file `configs/data_preparation.yaml` or create your own configuration file and place it in the `configs` folder.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). This will copy the current code base from your local machine to the GCP machine so make sure any changes to the configuration file are saved before this step is run.
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To create a dataset, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `pipenv run python3 prepare_dataset.py --config-file configs/<config_file>`. 
1. Once dataset preparation has finished, you should see the folder `<gcp_bucket>/datasets/<dataset_ID>-<timestamp>` has been created and populated, where `<dataset_ID>` was defined in `configs/data_preparation.yaml`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

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
1. Either edit the configuration file `configs/inference_config.yaml` or create your own configuration file and place it in the `configs` folder.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). This will copy the current code base from your local machine to the GCP machine so make sure any changes to the configuration file are saved before this step is run.
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To infer (segment) the damage of the stacks, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `pipenv run python3 infer_segmentation.py --gcp-bucket <gcp_bucket> --stack-id <stack_id> --model-id <model_id>`. 
1. Once inference has finished, you should see the folder `<gcp_bucket>/inferences/<inference_ID>-<timestamp>` has been created and populated, where `<inference_ID>` is `<stack_id>_<model_id>_<datetime>`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

## Testing a segmentation model

Prerequisite artifacts:
* A dataset (in a GCP bucket) that we will use to test the performance of the model against 
* A pretrained damage segmentation model (in a GCP bucket) to test the performance of
* A test configuration files (on your local machine)

Infrastructure that will be used:
* A GCP bucket where the segmented stacks will be accessed from
* A GCP bucket where the results of the analysis will be stored
* A GCP virtual machine to run the test on

### Workflow
1. If the stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Either edit the configuration file `configs/test_config.yaml` or create your own configuration file and place it in the `configs` folder.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). This will copy the current code base from your local machine to the GCP machine so make sure any changes to the configuration file are saved before this step is run.
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To create a dataset, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `pipenv run python3 test_segmentation_model.py --gcp-bucket <gcp_bucket> --dataset-id <dataset_id> --model-id <model_id>`.
1. Once dataset preparation has finished, you should see the folder `<gcp_bucket>/tests/<test_ID>-<timestamp>` has been created and populated, where `<test_ID>`  is `<dataset_id>_<model_id>_<datetime>`.
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
1. Either edit the configuration file `configs/train_config.yaml` or create your own configuration file and place it in the `configs` folder.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). This will copy the current code base from your local machine to the GCP machine so make sure any changes to the configuration file are saved before this step is run.
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To create a dataset, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `pipenv run python3 train_segmentation_model.py --config-file configs/<config_file>`. 
1. Once dataset preparation has finished, you should see the folder `<gcp_bucket>/models/<model_ID>-<timestamp>` has been created and populated, where `<model_ID>` was defined in `configs/train.yaml`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed. 
