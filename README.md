# necstlab's Damage Segmentation

There are three main workflows that this repository supports: 
* segmenting the damage on a scan (often called inference)
* testing the performance of a pretrained damage segmentation model 
* training a damage segmentation model

## Prerequisites for all workflows

All of the workflows use Google Cloud Platform (GCP) for storage (buckets) and compute (virtual machines). To allow the code to programmatically interact with GCP, we will set up a Software Development Kit (SDK) on your local machine. To install the GCP SDK follow the instructions [here](). TODO: add link/link content

To programmatically set up and destroy virtual machines, we will use a tool called Terraform. For instructions on how to install Terraform, see [here](). TODO: add link/link content

To set up and destroy virtual machines, Terraform requires access to GCP. For instructions on how to download GCP credentials for Terraform, see [here](). TODO: add link/link content

To store all of the artifacts (data, models, results, etc.) that are required for and result from the workflows, we will use a GCP bucket. To set up a GCP bucket to be used for these workflows, see the instructions [here](). TODO: add link/link content

## Segmenting damage of a scan

Prerequisite artifacts:
* Unsegmented scans that we wish to perform damage segmentation on
* A pretrained damage segmentation model
* An inference configuration file

Infrastructure that will be used:
* A GCP bucket where the unsegmented scans will be stored
* A GCP bucket where the segmented scans will be stored
* An appropriate GCP virtual machine to run on

Damage segmentation workflow:
* If the unsegmented scans are not in a GCP bucket, copy the unsegmented scans to a GCP bucket using your local machine
* Create an appropriate GCP virtual machine to use for the damage segmentation
* Use the GCP virtual machine to perform the damage segmentation on the scans
* Terminate the GCP virtual machine

### Copying the unsegmented scans to a GCP bucket

TODO

### Creating an appropriate GCP virtual machine

TODO

### Performing damage segmentation on the scans

TODO

### Terminating the GCP virtual machine

TODO

## Testing the segmentation model

Prerequisite artifacts:
* Segmented scans that we will use to test the performance of the model against
* A pretrained damage segmentation model to test the performance of
* A test configuration files

Infrastructure that will be used:
* A GCP bucket where the segmented scans will be stored
* A GCP bucket where the results of the analysis will be stored
* An appropriate GCP virtual machine to run the analysis on

Segmentation model testing workflow:
* If the segmented scans are not in a GCP bucket, copy the segmented scans to a GCP bucket using your local machine
* Create an appropriate GCP virtual machine to use for the analysis
* Use the GCP virtual machine to test the damage segmentation performance
* Terminate the GCP virtual machine

### Copying the segmented scans to a GCP bucket

TODO

### Creating an appropriate GCP virtual machine

TODO

### Testing the damage segmentation model

TODO

### Terminating the GCP virtual machine

TODO

## Training a segmentation model

Prerequisite artifacts:
* Segmented scans that we will use to train the model
* A train configuration file

Infrastructure that will be used:
* A GCP bucket where the segmented scans will be stored
* A GCP bucket where the prepared dataset for training will be stored
* A GCP bucket where the trained model will be stored
* An appropriate GCP virtual machine to run the training on

Segmentation model training workflow:
* If the segmented scans are not in a GCP bucket, copy the segmented scans to a GCP bucket using your local machine
* Create an appropriate GCP virtual machine to use for the data preparation and training
* Use the GCP virtual machine to prepare a dataset for training
* Use the GCP virtual machine to train a damage segmentation model
* Terminate the GCP virtual machine

### Copying the segmented scans to a GCP bucket

TODO

### Creating an appropriate GCP virtual machine

TODO

### Preparing a dataset to be used for training

TODO

### Training a damage segmentation model

TODO

### Terminating the GCP virtual machine

TODO

