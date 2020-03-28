# Data Ingestion

Prerequisite artifacts:
* A stack of (zipped) images or annotations (.tif's) that we wish to use in the other workflows on your local machine
* The zip filename is expected to look like:
    ``` 
    <stack_id>.zip (for unsegmented images)
    <stack_id>-dmg_labels_GV.zip (for annotations)    
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
1. SSH into the GCP virtual machine, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and process a single zip file by running the command: `python3 ingest_raw_data.py --gcp-bucket gs://<gcp_bucket_name> --zipped-stack gs://<gcp_bucket_name>/raw-data/<zip_file>`. Alternatively, to process an entire folder of zipped stacks, use `python3 ingest_raw_data.py --gcp-bucket gs://<gcp_bucket_name>` (excluding the `--zipped-stack` argument), which will process all of the files in `gs://<gcp_bucket_name>/raw-data` (`ingest_raw_data.py` knows to process only `<gcp_bucket_name>/raw-data`).
1. When this completes, you should see your stack in `gs://<gcp_bucket_name>/processed-data/<stack_ID>`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

### Summary of command line arguments of `ingest_raw_data.py`:

* `--gcp-bucket`:
        type=str,
        help='The GCP bucket where the raw data is located and to use to store the processed stacks.'
* `--zipped-stack`:
        type=str,
        default='',
        help='The zipped stack (.zip or .7z) to be processed.'
        
### Example command line inputs:

```
python3 ingest_raw_data.py --gcp-bucket gs://sandbox --zipped-stack gs://sandbox/raw-data/composite_0123.zip

python3 ingest_raw_data.py --gcp-bucket gs://sandbox --zipped-stack gs://sandbox/raw-data/composite_1234.7z

python3 ingest_raw_data.py --gcp-bucket gs://sandbox
```

### Notes:

- `ingest_raw_data.py` assumes that if `dmg` appears in the zip filename, then that the zip file has annotations. If no `dmg` appears, then it assumes it contains images.
- An identical `<stack_ID>` (image and/or annotations) existing in both `raw-data` and `processed-data` will be skipped by `ingest_raw_data.py`.

### Tips:

- In VM SSH, use `nano` text editor to edit scripts previously uploaded to VM. _E.g.,_ `nano configs/dataset-medium.yaml` to edit text in `dataset-medium.yaml`
- To create a VM without destroying others (assuming `terraform apply` seeks to create & destroy), use `target` flag: `terraform apply -lock=false -target=google_compute_instance.vm[<#>]` to create VM #. Similar syntax with `terraform destroy` to specify target.


