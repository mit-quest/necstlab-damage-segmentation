# Training

Prerequisite artifacts:
* A dataset (in a GCP bucket) that we will use to train the model
* A train configuration file (on your local machine)

Infrastructure that will be used:
* A GCP bucket where the prepared dataset for training will be accessed from
* A GCP bucket where the trained model will be stored
* A GCP virtual machine to run the training on

## Workflow
1. If the stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Either edit the configuration file `configs/train_config.yaml` or create your own configuration file and place it in the `configs` folder.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). This will copy the current code base from your local machine to the GCP machine so make sure any changes to the configuration file are saved before this step is run.
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To create a dataset, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `pipenv run python3 train_segmentation_model.py  --gcp-bucket <gcp_bucket> --config-file configs/<config_filename>.yaml`. 
1. Once dataset preparation has finished, you should see the folder `<gcp_bucket>/models/<model_ID>-<timestamp>` has been created and populated, where `<model_ID>` was defined in `configs/train_config.yaml`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed. 

## Notes
* Batch size of 16 works with P100 GPU, but batch size of 24 is too large.
