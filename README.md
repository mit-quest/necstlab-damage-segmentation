# necstlab's Damage Segmentation

This repository contains the instructions and code to train and use a model for damage segmentation of carbon-fiber-laminate scans. To accomplish this, there are five workflows that this repository supports: 
* copying the raw data into the cloud for storage and usage
* segmenting the damage on a scan (often called inference)
* creating a dataset for training and testing of a damage segmentation model
* testing the performance of a pretrained damage segmentation model on a dataset
* training a damage segmentation model on a dataset

## Prerequisites for all workflows

### Setting up your local machine

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

As you run the workflows you'll see the following directory structure be automatically created inside of your bucket:
```
<GCP bucket>
    \analyses     (this is where any analysis that results from testing will be stored)
        \<analysis_ID>-<timestamp>
            \plots
            config.yaml
            log.txt
            metrics.csv
    \raw-data     (this is where any raw data will be stored)
        \<scan_ID>
            \<annotations>
            \<images>
            metadata.yaml
    \datasets     (this is where any prepared datasets for training will be stored)
        \<dataset_ID>-<timestamp>
            \test
            \train
            \validate
            config.yaml
            log.txt
    \inferences   (this is where any scan segmentations will be stored)
        \<inference_ID>-<timestamp>
            \output
            config.yaml
            log.txt
    \models       (this is where any trained segmentation models will be stored)
        \<model_ID>-<timestamp>
            config.yaml
            log.txt
            model.hdf5
```
where `<analysis_ID>`, `<dataset_ID>`, `<inference_ID>`, and `<model_ID>` are defined inside of configuration files and `<timestamp>` is the time the process was started and is automatically generated. The `<scan_ID>`s will be created as raw data is moved into the cloud. 


### Assumed knowledge

The following workflows assume:
* You know how to check the status of GCP virtual machines using the GCP compute engine dashboard. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content 
* You know how to SSH into a GCP virtual machine. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content
* You know how to check the contents of a GCP bucket using the GCP storage dashboard. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content 
* You know how to create and destroy resources using Terraform. If you are unfamiliar with how to do this, see [here]() for instructions. TODO: add link/link content 

## Copying the raw data into the cloud for storage and usage

Prerequisite artifacts:
* Scans that we wish to use in the other workflows

Infrastructure that will be used:
* A GCP bucket where the scans will be stored
* Your local machine to upload the scans to the GCP bucket

Copying data to the cloud workflow:
* 

### Copying the unsegmented scans to a GCP bucket

To copy the unsegmented scans to a GCP bucket run the command:

`pipenv run python3 ingest_data_to_gcp.py --local-scans-dir <local_scans_dir> --gcp-bucket <gcp_bucket>`

where `<local_scans_dir>` is the local directory where the scans are stored and `<gcp_bucket>` is the bucket where our artifacts will be stored. 

When this completes, you should see all of your scans in `<gcp_bucket>/data/ingested/<scan_ID>` where `<scan_ID>` are the names of the directories inside of `<local_scans_dir>`.

## Segmenting damage of a scan

Prerequisite artifacts:
* Unsegmented scans that we wish to perform damage segmentation on in a GCP bucket
* A pretrained damage segmentation model in a GCP bucket
* An inference configuration file

Infrastructure that will be used:
* A GCP bucket where the stored unsegmented scans will be accessed from
* A GCP bucket where the segmented scans will be stored
* An appropriate GCP virtual machine to run on

Damage segmentation workflow:
* Ensuring the unsegmented scans are in a GCP bucket
* Use a GCP virtual machine to perform the damage segmentation on the scans

### Copying the unsegmented scans to a GCP bucket

If the unsegmented scans are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.

### Performing damage segmentation on the scans

Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.

To infer (segment) the damage of the scans SSH into the virtual machine `<project_name>-<user_name>` and run `pipenv run python3 infer_segmentation.py --config-file configurations/inference.yaml`. Once inference has finished, you should see the folder `<gcp_bucket>/inferences/<inference_ID>-<timestamp>` has been created and populated, where `<inference_ID>` was defined in `configurations/inference.yaml`.

Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

## Testing the segmentation model

Prerequisite artifacts:
* Segmented scans that we will use to test the performance of the model against
* A pretrained damage segmentation model to test the performance of
* A test configuration files

Infrastructure that will be used:
* A GCP bucket where the segmented scans will be accessed from
* A GCP bucket where the results of the analysis will be stored
* An appropriate GCP virtual machine to run the analysis on

Segmentation model testing workflow:
* Ensuring the segmented scans are in a GCP bucket
* Use a GCP virtual machine to test the damage segmentation performance

### Copying the segmented scans to a GCP bucket

If the segmented scans are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.

### Testing the damage segmentation model

TODO

## Training a segmentation model

Prerequisite artifacts:
* Segmented scans that we will use to train the model
* A train configuration file

Infrastructure that will be used:
* A GCP bucket where the segmented scans will be accessed from
* A GCP bucket where the prepared dataset for training will be stored
* A GCP bucket where the trained model will be stored
* An appropriate GCP virtual machine to run the training on

Segmentation model training workflow:
* Ensuring the segmented scans are in a GCP bucket
* Use a GCP virtual machine to prepare a dataset for training
* Use a GCP virtual machine to train a damage segmentation model

### Copying the segmented scans to a GCP bucket

If the segmented scans are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.

### Preparing a dataset to be used for training

TODO

### Training a damage segmentation model

TODO
