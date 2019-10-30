# Dataset Preparation

Prerequisite artifacts:
* Annotated stacks (in a GCP bucket) that we will use to create the dataset
* A dataset configuration file (on your local machine)

Infrastructure that will be used:
* A GCP bucket where the segmented stacks will be accessed from
* A GCP bucket where the prepared dataset will be stored
* A GCP virtual machine to run the training on

## Workflow

1. If the stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Either edit the configuration file `configs/data_preparation.yaml` or create your own configuration file and place it in the `configs` folder.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). This will copy the current code base from your local machine to the GCP machine so make sure any changes to the configuration file are saved before this step is run.
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To create a dataset, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `pipenv run python3 prepare_dataset.py --config-file configs/<config_file>`. 
1. Once dataset preparation has finished, you should see the folder `<gcp_bucket>/datasets/<dataset_ID>` has been created and populated, where `<dataset_ID>` was defined in `configs/data_preparation.yaml`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.
