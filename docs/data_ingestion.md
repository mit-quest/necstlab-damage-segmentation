# Data Ingestion

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

## Workflow

1. Copy the zip files of raw data into a GCP bucket: `gsutil -m cp <local_data_file> gs://<gcp_bucket_name>/raw-data` where `<gcp_bucket_name>` is the bucket where our artifacts will be stored. To copy an entire folder of zip files, `cd` into the directory and use the command: `gsutil -m cp -r . gs://<gcp_bucket_name>/raw-data`
1. When this completes, you should see your stack in `gs://<gcp_bucket_name>/raw-data/<zip_file>`.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply` or `terraform apply -lock=false`). 
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. SSH into the GCP virtual machine, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and process a zip file by running the command: `pipenv run python3 ingest_raw_data.py --gcp-bucket gs://<gcp_bucket_name> --zipped-stack gs://<gcp_bucket_name>/raw-data/<zip_file>`. If `pipenv run python3 ingest_raw_data.py --gcp-bucket gs://<gcp_bucket_name>` (excluding the `--zipped-stack` argument) it will process all of the files in `gs://<gcp_bucket_name>/raw-data`.
1. When this completes, you should see your stack in `gs://<gcp_bucket_name>/processed-data/<stack_ID>`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

Note:
- `ingest_raw_data.py` assumes that if `dmg` appears in the zip filename, then that the zip file has annotations. If no `dmg` appears, then it assumes it contains images.
