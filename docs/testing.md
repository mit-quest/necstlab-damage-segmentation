# Testing

Prerequisite artifacts:
* A dataset (in a GCP bucket) that we will use to test the performance of the model against 
* A pretrained damage segmentation model (in a GCP bucket) to test the performance of

Infrastructure that will be used:
* A GCP bucket where the segmented stacks will be accessed from
* A GCP bucket where the results of the analysis will be stored
* A GCP virtual machine to run the test on

## Workflow
1. If the stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`).
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To create a dataset, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `pipenv run python3 test_segmentation_model.py --gcp-bucket <gcp_bucket> --dataset-id <dataset_id> --model-id <model_id>`.
1. Once dataset preparation has finished, you should see the folder `<gcp_bucket>/tests/<test_ID>` has been created and populated, where `<test_ID>`  is `<dataset_id>_<model_id>_<datetime>`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed. 
