# Inference

Prerequisite artifacts:
* Unannotated stacks (in a GCP bucket) that we wish to perform damage segmentation on
* A pretrained damage segmentation model (in a GCP bucket) to use for inference

Infrastructure that will be used:
* A GCP bucket where the stored unsegmented stacks will be accessed from
* A GCP bucket where the stacks with inferred damage segmentation will be stored
* A GCP virtual machine to run inference on

## Workflow

1. If the unsegmented stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`).
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To infer (segment) the damage of the stacks, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `pipenv run python3 infer_segmentation.py --gcp-bucket <gcp_bucket> --stack-id <stack_id> --model-id <model_id>`. 
1. Once inference has finished, you should see the folder `<gcp_bucket>/inferences/<inference_ID>` has been created and populated, where `<inference_ID>` is `<stack_id>_<model_id>`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

## Notes
- `infer_segmentation.py` looks for `<stack_id>` in `processed-data` inside bucket
- Download inference locally by (e.g.) `gsutil -m cp -r gs://necstlab-sandbox/inferences/<inference_ID> <local_storage_location>`
