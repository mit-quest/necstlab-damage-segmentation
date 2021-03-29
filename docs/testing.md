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
1. To test a model, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `python3 test_segmentation_model.py --gcp-bucket <gcp_bucket> --dataset-id <dataset_id> --model-id <model_id>`. _Optional:_ Use `--trained-thresholds-id <model_thresholds>.yaml` to test using pretrained class thresholds.
1. Once testing has finished, you should see the folder `<gcp_bucket>/tests/<test_ID>` has been created and populated, where `<test_ID>`  is `<dataset_id>_<model_id>`. Since testing can be performed multiple times using different prediction thresholds, output `metadata.yaml` and `metrics.csv` filenames are appended with `_<timestamp>`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed. 

### Summary of command line arguments of `test_segmentation_model.py`:

* `--gcp-bucket`:
        type=str,
        help='The GCP bucket where the raw data is located and to use to store the processed stacks.'
* `--dataset-id`:
        type=str,
        help='The dataset ID.'
* `--model-id`:
        type=str,
        help='The model ID.'
* `--batch-size`:
        type=int,
        default=16,
        help='The batch size to use during inference.'
* `--trained-thresholds-id`:
        type=str,
        default=None,
        help='The specified trained thresholds file id.'
        
### Example command line inputs:

```
python3 test_segmentation_model.py --gcp-bucket gs://sandbox --dataset-id dataset-composite_0123 --model-id segmentation-model-composite_0123_20200321T154533Z --batch-size 16

python3 test_segmentation_model.py --gcp-bucket gs://sandbox --dataset-id dataset-composite_0123 --model-id segmentation-model-composite_0123_20200321T154533Z --batch-size 16 --trained-thresholds-id model_thresholds_20200321T181016Z.yaml
``` 

### Notes:

- Batch size of 16 works with P100 GPU (8 for K80), but batch size of 20 is too large for P100 GPU.
- Metrics will be computed based on `global_threshold` in `metrics_utils.py` unless pretrained thresholds are specified.

### Tips:

- In VM SSH, use `nano` text editor to edit scripts previously uploaded to VM. _E.g.,_ `nano configs/dataset-medium.yaml` to edit text in `dataset-medium.yaml`
- To create a VM without destroying others (assuming `terraform apply` seeks to create & destroy), use `target` flag: `terraform apply -lock=false -target=google_compute_instance.vm[<#>]` to create VM #. Similar syntax with `terraform destroy` to specify target. 
